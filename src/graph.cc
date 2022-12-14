#include "./graph.h"

#include <sstream>
#include "./bcast.h"
#include "./cuda/sddmm.h"
#include "./graph_ops.h"

namespace gs {

void Graph::LoadCSC(torch::Tensor indptr, torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->indptr = indptr;
  csc_->indices = indices;
  num_cols_ = indptr.numel() - 1;
  num_rows_ = indptr.numel() - 1;
  num_edges_ = indices.numel();
  LOG(INFO) << "Loaded CSC with " << indptr.numel() - 1 << " nodes and "
            << indices.numel() << " edges";
}

void Graph::LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr,
                              torch::Tensor indices) {
  if (column_ids.numel() != indptr.numel() - 1) {
    LOG(FATAL) << "length of column_ids is not aligned with indptr";
  }
  csc_ = std::make_shared<CSC>();
  col_ids_ = column_ids;
  csc_->indptr = indptr;
  csc_->indices = indices;
  num_cols_ = column_ids.numel();
  num_rows_ = indices.max().item<int64_t>() + 1;
  num_edges_ = indices.numel();
}

std::shared_ptr<CSC> Graph::GetCSC() {
  if (csc_ == nullptr) {
    if (coo_ != nullptr) {
      SetCSC(GraphCOO2CSC(coo_, num_cols_, true));
    } else if (csr_ != nullptr) {
      CSR2CSC();
    }
  }
  return csc_;
}

std::shared_ptr<CSR> Graph::GetCSR() {
  if (csr_ == nullptr) {
    if (coo_ != nullptr) {
      SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
    } else if (csc_ != nullptr) {
      CSC2CSR();
    }
  }
  return csr_;
}

std::shared_ptr<COO> Graph::GetCOO() {
  if (coo_ == nullptr) {
    if (csc_ != nullptr) {
      SetCOO(GraphCSC2COO(csc_, true));
    } else if (csr_ != nullptr) {
      SetCOO(GraphCSC2COO(csr_, false));
    }
  }
  return coo_;
}

torch::optional<torch::Tensor> Graph::GetData(std::string order) {
  auto data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  bool need_idx = false;
  torch::Tensor idx;
  if (order == "col") {
    if (csc_ == nullptr) {
      GetCSC();
    }
    if (csc_->e_ids.has_value()) {
      need_idx = true;
      idx = csc_->e_ids.value();
    }
  } else if (order == "row") {
    if (csr_ == nullptr) {
      GetCSR();
    }
    if (csr_->e_ids.has_value()) {
      need_idx = true;
      idx = csr_->e_ids.value();
    }
  }
  return need_idx ? data[idx] : data;
}

int64_t Graph::GetNumCols() { return num_cols_; }

int64_t Graph::GetNumRows() { return num_rows_; }

int64_t Graph::GetNumEdges() { return num_edges_; }

void Graph::SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }

void Graph::SetCSR(std::shared_ptr<CSR> csr) { csr_ = csr; }

void Graph::SetCOO(std::shared_ptr<COO> coo) { coo_ = coo; }

void Graph::SetData(torch::Tensor data, std::string order) {
  bool need_permutation = false;
  torch::Tensor idx;
  if (order == "col") {
    GetCSC();
    need_permutation = csc_->e_ids.has_value();
    if (need_permutation) {
      idx = csc_->e_ids.value();
    }
  } else if (order == "row") {
    GetCSR();
    need_permutation = csr_->e_ids.has_value();
    if (need_permutation) {
      idx = csr_->e_ids.value();
    }
  }
  data_ = need_permutation ? data[idx] : data;
}

c10::intrusive_ptr<Graph> Graph::FusedBidirSlicing(torch::Tensor column_seeds,
                                                   torch::Tensor row_seeds) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_seeds})
                              : column_seeds;
  torch::Tensor row_ids =
      (row_ids_.has_value()) ? row_ids_.value().index({row_seeds}) : row_seeds;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(
      true, col_ids, row_ids, column_seeds.numel(), row_seeds.numel())));
  std::tie(csc_ptr, select_index) =
      FusedCSCColRowSlicing(csc_, col_ids, row_ids);
  ret->SetCSC(csc_ptr);
  if (data_.has_value()) {
    if (csc_->e_ids.has_value()) {
      out_data =
          data_.value().index({csc_->e_ids.value().index({select_index})});
    } else {
      out_data = data_.value().index({select_index});
    }
    ret->SetData(out_data);
  }
  return ret;
}

void Graph::SetNumEdges(int64_t num_edges) { num_edges_ = num_edges; }

c10::intrusive_ptr<Graph> Graph::ColumnwiseSlicing(torch::Tensor column_index) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_index})
                              : column_index;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids, row_ids_, column_index.numel(), num_rows_)));
  std::tie(csc_ptr, select_index) = CSCColSlicing(csc_, column_index);
  ret->SetCSC(csc_ptr);
  ret->SetNumEdges(csc_ptr->indices.numel());
  if (data_.has_value()) {
    if (csc_->e_ids.has_value()) {
      out_data =
          data_.value().index({csc_->e_ids.value().index({select_index})});
    } else {
      out_data = data_.value().index({select_index});
    }
    ret->SetData(out_data);
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::RowwiseSlicing(torch::Tensor row_index) {
  torch::Tensor select_index, out_data;
  torch::Tensor row_ids =
      (row_ids_.has_value()) ? row_ids_.value().index({row_index}) : row_index;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids, num_cols_, row_index.numel())));
  if (csr_ != nullptr) {
    std::shared_ptr<CSR> csr_ptr;
    std::tie(csr_ptr, select_index) = CSCColSlicing(csr_, row_index);
    ret->SetCSR(csr_ptr);
    ret->SetNumEdges(csr_ptr->indices.numel());
    if (data_.has_value()) {
      if (csr_->e_ids.has_value()) {
        out_data =
            data_.value().index({csr_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }
  } else if (csc_ != nullptr) {
    std::shared_ptr<CSC> csc_ptr;
    std::tie(csc_ptr, select_index) = CSCRowSlicing(csc_, row_index);
    ret->SetCSC(csc_ptr);
    ret->SetNumEdges(csc_ptr->indices.numel());
    if (data_.has_value()) {
      if (csc_->e_ids.has_value()) {
        out_data =
            data_.value().index({csc_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }
  } else {
    LOG(FATAL) << "Error in RowwiseSlicing: no CSC nor CSR";
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::ColumnwiseSampling(int64_t fanout,
                                                    bool replace) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids_, num_cols_, num_rows_)));
  std::tie(csc_ptr, select_index) = CSCColSampling(csc_, fanout, replace);
  ret->SetCSC(csc_ptr);
  ret->SetNumEdges(csc_ptr->indices.numel());
  if (data_.has_value()) {
    if (csc_->e_ids.has_value()) {
      out_data =
          data_.value().index({csc_->e_ids.value().index({select_index})});
    } else {
      out_data = data_.value().index({select_index});
    }
    ret->SetData(out_data);
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::ColumnwiseSamplingProbs(
    torch::Tensor edge_probs, int64_t fanout, bool replace) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids_, num_cols_, num_rows_)));
  std::tie(csc_ptr, select_index) =
      CSCColSamplingProbs(csc_, edge_probs, fanout, replace);
  ret->SetCSC(csc_ptr);
  ret->SetNumEdges(csc_ptr->indices.numel());
  if (data_.has_value()) {
    if (csc_->e_ids.has_value()) {
      out_data =
          data_.value().index({csc_->e_ids.value().index({select_index})});
    } else {
      out_data = data_.value().index({select_index});
    }
    ret->SetData(out_data);
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::ColumnwiseFusedSlicingAndSampling(
    torch::Tensor column_index, int64_t fanout, bool replace) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_index})
                              : column_index;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids, row_ids_, num_cols_, num_rows_)));
  std::tie(csc_ptr, select_index) =
      FusedCSCColSlicingAndSampling(csc_, column_index, fanout, replace);
  ret->SetCSC(csc_ptr);
  ret->SetNumEdges(csc_ptr->indices.numel());
  if (data_.has_value()) {
    if (csc_->e_ids.has_value()) {
      out_data =
          data_.value().index({csc_->e_ids.value().index({select_index})});
    } else {
      out_data = data_.value().index({select_index});
    }
    ret->SetData(out_data);
  }
  return ret;
}

void Graph::CSC2CSR() {
  SetCOO(GraphCSC2COO(csc_, true));
  SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
}

void Graph::CSR2CSC() {
  SetCOO(GraphCSC2COO(csr_, false));
  SetCSC(GraphCOO2CSC(coo_, num_cols_, true));
}

void Graph::CreateSparseFormat(int64_t axis) {
  if (axis != 0 && axis != 1) {
    LOG(FATAL) << "axis should be 0 or 1";
  }
  if (axis == 0 && csc_ == nullptr) {
    CSR2CSC();
  } else if (axis == 1 && csr_ == nullptr) {
    CSC2CSR();
  }
}

torch::Tensor Graph::RandomWalk(torch::Tensor seeds, int64_t walk_length) {
  return FusedRandomWalk(this->csc_, seeds, walk_length);
}

torch::Tensor Graph::Sum(int64_t axis, int64_t powk) {
  torch::Tensor out_data;
  CreateSparseFormat(axis);
  if (axis == 0) {
    out_data = GraphSum(csc_, GetData(), powk);
  } else if (axis == 1) {
    out_data = GraphSum(csr_, GetData(), powk);
  }
  return out_data;
}

c10::intrusive_ptr<Graph> Graph::Divide(torch::Tensor divisor, int64_t axis) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  torch::Tensor out_data;
  CreateSparseFormat(axis);
  if (axis == 0) {
    out_data = GraphDiv(csc_, GetData(), divisor);
  } else if (axis == 1) {
    out_data = GraphDiv(csr_, GetData(), divisor);
  }
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  ret->SetNumEdges(num_edges_);
  ret->SetData(out_data);
  return ret;
}

c10::intrusive_ptr<Graph> Graph::Normalize(int64_t axis) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  torch::Tensor out_data;
  CreateSparseFormat(axis);
  if (axis == 0) {
    out_data = GraphNormalize(csc_, GetData());
  } else if (axis == 1) {
    out_data = GraphNormalize(csr_, GetData());
  }
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  ret->SetNumEdges(num_edges_);
  ret->SetData(out_data);
  return ret;
}

torch::Tensor Graph::AllValidNode() {
  // for sampling, col_ids_ is necessary!
  if (!col_ids_.has_value()) {
    LOG(ERROR) << "For sampling, col_ids_ is necessary!";
  }

  torch::Tensor col_ids = col_ids_.value();
  torch::Tensor row_ids;
  if (csc_ != nullptr) {
    row_ids = (row_ids_.has_value()) ? row_ids_.value() : csc_->indices;
  } else if (coo_ != nullptr) {
    row_ids = (row_ids_.has_value()) ? row_ids_.value() : coo_->row;
  } else if (csr_ != nullptr) {
    // turn csr2coo
    SetCOO(GraphCSC2COO(csr_, false));
    row_ids = (row_ids_.has_value()) ? row_ids_.value() : coo_->row;
  } else {
    LOG(ERROR) << "Error in AllValidNode!";
  }

  return TensorUnique(torch::cat({col_ids, row_ids}));
}

std::tuple<torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor,
           torch::optional<torch::Tensor>, std::string>
Graph::Relabel() {
  // for sampling, col_ids_ is necessary!
  if (!col_ids_.has_value()) {
    LOG(ERROR) << "For sampling, col_ids_ is necessary!";
  }
  torch::Tensor col_ids = col_ids_.value();

  if (csc_ != nullptr) {
    torch::Tensor row_ids = (row_ids_.has_value())
                                ? row_ids_.value().index({csc_->indices})
                                : csc_->indices;
    torch::Tensor row_indices = row_ids_.has_value()
                                    ? row_ids_.value().index({csc_->indices})
                                    : csc_->indices;

    torch::Tensor frontier;
    std::vector<torch::Tensor> relabeled_result;

    std::tie(frontier, relabeled_result) =
        BatchTensorRelabel({col_ids, row_ids}, {row_indices});

    torch::Tensor relabeled_indptr = csc_->indptr.clone();
    torch::Tensor relabeled_indices = relabeled_result[0];

    return {frontier,
            frontier.numel(),
            col_ids.numel(),
            relabeled_indptr,
            relabeled_indices,
            csc_->e_ids,
            "csc"};

  } else if (csr_ != nullptr or coo_ != nullptr) {
    if (coo_ == nullptr) {
      SetCOO(GraphCSC2COO(csr_, false));
    }
    torch::Tensor row_ids = (row_ids_.has_value())
                                ? row_ids_.value().index({coo_->row})
                                : coo_->row;

    torch::Tensor coo_col = col_ids.index({coo_->col});
    torch::Tensor coo_row = (row_ids_.has_value())
                                ? row_ids_.value().index({coo_->row})
                                : coo_->row;

    torch::Tensor frontier;
    std::vector<torch::Tensor> relabeled_result;
    std::tie(frontier, relabeled_result) =
        BatchTensorRelabel({col_ids, row_ids}, {coo_col, coo_row});

    return {frontier,
            frontier.numel(),
            col_ids.numel(),
            relabeled_result[1],
            relabeled_result[0],
            coo_->e_ids,
            "coo"};

  } else {
    LOG(ERROR) << "Error in Relabel!";
    return {};
  }
}

torch::Tensor Graph::GetRows() {
  if (row_ids_.has_value()) {
    return row_ids_.value();
  } else {
    if (csr_ != nullptr) {
      return torch::arange(num_rows_, csr_->indices.options());
    } else {
      return torch::arange(num_rows_, csc_->indices.options());
    }
  }
}

torch::Tensor Graph::GetCols() {
  if (col_ids_.has_value()) {
    return col_ids_.value();
  } else {
    if (csc_ != nullptr) {
      return torch::arange(num_cols_, csc_->indices.options());
    } else {
      return torch::arange(num_cols_, csr_->indices.options());
    }
  }
}

torch::Tensor Graph::GetValidRows() {
  if (row_ids_.has_value()) {
    return row_ids_.value();
  } else {
    return std::get<0>(torch::_unique(GetCOORows(true)));
  }
}

torch::Tensor Graph::GetValidCols() {
  if (col_ids_.has_value()) {
    return col_ids_.value();
  } else {
    return std::get<0>(torch::_unique(GetCOOCols(true)));
  }
}

torch::Tensor Graph::GetCOORows(bool is_original) {
  torch::Tensor coo_rows;
  if (coo_ != nullptr) {
    coo_rows = coo_->row;
  } else if (csc_ != nullptr && !csc_->e_ids.has_value()) {
    coo_rows = csc_->indices;
  } else if (csr_ != nullptr && !csr_->e_ids.has_value()) {
    SetCOO(GraphCSC2COO(csr_, false));
    coo_rows = coo_->row;
  } else {
    LOG(ERROR) << "Error in GetCOORows!";
  }

  return (is_original && row_ids_.has_value())
             ? row_ids_.value().index({coo_rows})
             : coo_rows;
};

torch::Tensor Graph::GetCOOCols(bool is_original) {
  torch::Tensor coo_cols;
  if (coo_ != nullptr) {
    coo_cols = coo_->col;
  } else if (csc_ != nullptr && !csc_->e_ids.has_value()) {
    SetCOO(GraphCSC2COO(csc_, true));
    coo_cols = coo_->col;
  } else if (csr_ != nullptr && !csr_->e_ids.has_value()) {
    coo_cols = csr_->indices;
  } else {
    LOG(ERROR) << "Error in GetCOOCols!";
  }
  return (is_original && col_ids_.has_value())
             ? col_ids_.value().index({coo_cols})
             : coo_cols;
};

std::vector<torch::Tensor> Graph::MetaData() {
  torch::Tensor col_ids, row_ids, data;
  col_ids = (col_ids_.has_value()) ? col_ids_.value() : torch::Tensor();
  row_ids = (row_ids_.has_value()) ? row_ids_.value() : torch::Tensor();
  if (csc_ != nullptr) {
    data =
        (data_.has_value())
            ? data_.value()
            : torch::ones(csc_->indices.numel(),
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return {col_ids, row_ids, data, csc_->indptr, csc_->indices};
  } else if (csr_ != nullptr) {
    data =
        (data_.has_value())
            ? data_.value()
            : torch::ones(csr_->indices.numel(),
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return {col_ids, row_ids, data, csr_->indptr, csr_->indices};
  } else {
    LOG(FATAL) << "Error in MetaData: no CSC nor CSR.";
    return {coo_->row, coo_->col};
  }
}

/*! \brief Generalized Sampled Dense-Dense Matrix Multiplication. */
void Graph::SDDMM(const std::string& op, torch::Tensor lhs, torch::Tensor rhs,
                  torch::Tensor out, int64_t lhs_target, int64_t rhs_target) {
  const auto& bcast = CalcBcastOff(op, lhs, rhs);
  if (csc_ != nullptr) {
    auto csr_ptr =
        std::make_shared<CSR>(CSR{csc_->indptr, csc_->indices, csc_->e_ids});
    lhs_target = lhs_target == 1 ? lhs_target : (2 - lhs_target);
    rhs_target = rhs_target == 1 ? rhs_target : (2 - rhs_target);
    impl::SDDMMCSR(op, bcast, csr_ptr, lhs, rhs, out, lhs_target, rhs_target);
  } else if (csr_ != nullptr) {
    impl::SDDMMCSR(op, bcast, csr_, lhs, rhs, out, lhs_target, rhs_target);
  } else if (coo_ != nullptr) {
    impl::SDDMMCOO(op, bcast, coo_, lhs, rhs, out, lhs_target, rhs_target);
  } else {
    LOG(FATAL) << "SDDMM only supports CSR and COO formats";
  }
}

}  // namespace gs
