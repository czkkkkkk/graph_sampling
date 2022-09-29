#include "./graph.h"

#include <sstream>

#include "./graph_ops.h"

namespace gs {

void Graph::LoadCSC(torch::Tensor indptr, torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->indptr = indptr;
  csc_->indices = indices;
  num_cols_ = indptr.numel() - 1;
  num_rows_ = indptr.numel() - 1;
  LOG(INFO) << "Loaded CSC with " << indptr.numel() - 1 << " nodes and "
            << indices.numel() << " edges";
}

void Graph::LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr,
                              torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  col_ids_ = column_ids;
  csc_->indptr = indptr;
  csc_->indices = indices;
  num_cols_ = column_ids.numel();
  num_rows_ = indices.max().item<int64_t>() + 1;
}

std::shared_ptr<CSC> Graph::GetCSC() { return csc_; }

std::shared_ptr<CSR> Graph::GetCSR() { return csr_; }

std::shared_ptr<COO> Graph::GetCOO() { return coo_; }

torch::optional<torch::Tensor> Graph::GetData() { return data_; }

int64_t Graph::GetNumCols() { return num_cols_; }

int64_t Graph::GetNumRows() { return num_rows_; }

void Graph::SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }

void Graph::SetCSR(std::shared_ptr<CSR> csr) { csr_ = csr; }

void Graph::SetCOO(std::shared_ptr<COO> coo) { coo_ = coo; }

void Graph::SetData(torch::Tensor data) { data_ = data; }

// @todo data
c10::intrusive_ptr<Graph> Graph::ColumnwiseSlicing(torch::Tensor column_index) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_index})
                              : column_index;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids, row_ids_, column_index.numel(), num_rows_)));
  std::tie(csc_ptr, select_index) = CSCColumnwiseSlicing(csc_, column_index);
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

// @todo data
// @todo Fix for new storage format (index and duplicated rows)
c10::intrusive_ptr<Graph> Graph::RowwiseSlicing(torch::Tensor row_index) {
  torch::Tensor select_index, out_data;
  torch::Tensor row_ids =
      (row_ids_.has_value()) ? row_ids_.value().index({row_index}) : row_index;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids, num_cols_, row_index.numel())));
  if (csr_ != nullptr) {
    std::shared_ptr<CSR> csr_ptr;
    std::tie(csr_ptr, select_index) = CSRRowwiseSlicing(csr_, row_index);
    ret->SetCSR(csr_ptr);
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
    ret->SetCSC(CSCRowwiseSlicing(csc_, row_index));
  } else {
    LOG(FATAL) << "Error in RowwiseSlicing: no CSC nor CSR";
  }
  return ret;
}

// @todo data
c10::intrusive_ptr<Graph> Graph::ColumnwiseSampling(int64_t fanout,
                                                    bool replace) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids_, num_cols_, num_rows_)));
  std::tie(csc_ptr, select_index) =
      CSCColumnwiseSampling(csc_, fanout, replace);
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

// @todo data
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
      CSCColumnwiseFusedSlicingAndSampling(csc_, column_index, fanout, replace);
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

void Graph::CSC2CSR() {
  SetCOO(GraphCSC2COO(csc_));
  int64_t num_rows =
      (row_ids_.has_value()) ? row_ids_.value().numel() : num_rows_;
  SetCSR(GraphCOO2CSR(coo_, num_rows));
}

void Graph::CSR2CSC() {
  SetCOO(GraphCSR2COO(csr_));
  int64_t num_cols =
      (col_ids_.has_value()) ? col_ids_.value().numel() : num_cols_;
  SetCSC(GraphCOO2CSC(coo_, num_cols));
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>,
           torch::optional<torch::Tensor>>
Graph::PrepareDataForCompute(int64_t axis) {
  assertm(axis == 0 || axis == 1, "axis should be 0 or 1");
  torch::Tensor indptr;
  torch::optional<torch::Tensor> data, e_ids;
  data = GetData();
  if (axis == 0) {
    if (csc_ == nullptr) {
      CSR2CSC();
    }
    indptr = csc_->indptr;
    e_ids = csc_->e_ids;
  } else {
    if (csr_ == nullptr) {
      CSC2CSR();
    }
    indptr = csr_->indptr;
    e_ids = csr_->e_ids;
  }
  return {indptr, data, e_ids};
}

torch::Tensor Graph::Sum(int64_t axis) {
  torch::Tensor indptr, out_data;
  torch::optional<torch::Tensor> in_data, e_ids;
  std::tie(indptr, in_data, e_ids) = PrepareDataForCompute(axis);
  out_data = GraphSum(indptr, e_ids, in_data);
  return out_data;
}

torch::Tensor Graph::L2Norm(int64_t axis) {
  torch::Tensor indptr, out_data;
  torch::optional<torch::Tensor> in_data, e_ids;
  std::tie(indptr, in_data, e_ids) = PrepareDataForCompute(axis);
  out_data = GraphL2Norm(indptr, e_ids, in_data);
  return out_data;
}

c10::intrusive_ptr<Graph> Graph::Divide(torch::Tensor divisor, int64_t axis) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  torch::Tensor indptr, out_data;
  torch::optional<torch::Tensor> in_data, e_ids;
  std::tie(indptr, in_data, e_ids) = PrepareDataForCompute(axis);
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  out_data = GraphDiv(indptr, e_ids, in_data, divisor);
  ret->SetData(out_data);
  return ret;
}

c10::intrusive_ptr<Graph> Graph::Normalize(int64_t axis) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  torch::Tensor indptr, out_data;
  torch::optional<torch::Tensor> in_data, e_ids;
  std::tie(indptr, in_data, e_ids) = PrepareDataForCompute(axis);
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  out_data = GraphNormalize(indptr, e_ids, in_data);
  ret->SetData(out_data);
  return ret;
}

torch::Tensor Graph::RowIndices(bool unique) {
  torch::Tensor row_indices;
  if (row_ids_.has_value()) {
    row_indices = row_ids_.value();
    if (unique) {
      row_indices = std::get<0>(torch::_unique(row_indices));
    }
  } else {
    if (csr_ != nullptr) {
      row_indices =
          torch::arange(csr_->indptr.numel() - 1, csr_->indptr.options());
    } else if (csc_ != nullptr) {
      row_indices = std::get<0>(torch::_unique(csc_->indices));
    } else {
      LOG(FATAL) << "Error in RowIndices: no CSC nor CSR";
      row_indices = torch::Tensor();
    }
  }
  return row_indices;
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
    SetCOO(GraphCSR2COO(csr_));
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
    torch::Tensor row_ids =
        (row_ids_.has_value()) ? row_ids_.value() : csc_->indices;
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
      SetCOO(GraphCSR2COO(csr_));
    }
    torch::Tensor row_ids =
        (row_ids_.has_value()) ? row_ids_.value() : coo_->row;

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
  }
}

void Graph::Print() const {
  std::stringstream ss;
  if (col_ids_.has_value()) {
    ss << "col ids: " << col_ids_.value() << "\n";
  }
  if (row_ids_.has_value()) {
    ss << "col ids: " << row_ids_.value() << "\n";
  }
  if (csc_ != nullptr) {
    ss << "# Nodes: " << csc_->indptr.size(0) - 1
       << " # Edges: " << csc_->indices.size(0) << "\n";
    ss << "CSC indptr: "
       << "\n"
       << csc_->indptr << "\n";
    ss << "CSC indices: "
       << "\n"
       << csc_->indices << "\n";
  }
  if (csr_ != nullptr) {
    ss << "# Nodes: " << csr_->indptr.size(0) - 1
       << " # Edges: " << csr_->indices.size(0) << "\n";
    ss << "CSR indptr: "
       << "\n"
       << csr_->indptr << "\n";
    ss << "CSR indices: "
       << "\n"
       << csr_->indices << "\n";
  }
  LOG(INFO) << ss.str();
}

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

}  // namespace gs
