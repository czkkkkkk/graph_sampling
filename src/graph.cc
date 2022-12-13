#include "./graph.h"

#include <sstream>
#include "bcast.h"
#include "cuda/graph_ops.h"
#include "cuda/sddmm.h"
#include "cuda/tensor_ops.h"
#include "graph_ops.h"

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

std::shared_ptr<CSC> Graph::GetCSC() { return csc_; }

std::shared_ptr<CSR> Graph::GetCSR() { return csr_; }

std::shared_ptr<COO> Graph::GetCOO() { return coo_; }

torch::optional<torch::Tensor> Graph::GetData() { return data_; }

int64_t Graph::GetNumCols() { return num_cols_; }

int64_t Graph::GetNumRows() { return num_rows_; }

int64_t Graph::GetNumEdges() { return num_edges_; }

void Graph::SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }

void Graph::SetCSR(std::shared_ptr<CSR> csr) { csr_ = csr; }

void Graph::SetCOO(std::shared_ptr<COO> coo) { coo_ = coo; }

void Graph::SetData(torch::Tensor data) { data_ = data; }

void Graph::SetValidCols(torch::Tensor val_cols) { val_col_ids_ = val_cols; }

void Graph::SetValidRows(torch::Tensor val_rows) { val_row_ids_ = val_rows; }

void Graph::CreateCOO() {
  if (coo_ != nullptr) return;
  if (csc_ != nullptr) {
    if (val_col_ids_.has_value())
      SetCOO(GraphDCSC2COO(csc_, val_col_ids_.value(), true));
    else
      SetCOO(GraphCSC2COO(csc_, true));
  } else {
    if (val_row_ids_.has_value())
      SetCOO(GraphDCSC2COO(csr_, val_row_ids_.value(), false));
    else
      SetCOO(GraphCSC2COO(csr_, false));
  }
}

void Graph::CreateCSR() {
  if (csr_ != nullptr) return;
  if (coo_ != nullptr) {
    if (is_subgraph_) {
      val_row_ids_ = std::get<0>(torch::_unique(coo_->row));
      SetCSR(GraphCOO2DCSC(coo_, val_row_ids_.value(), false));
    } else {
      SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
    }
  } else {
    if (is_subgraph_) {
      CSC2DCSR();
    } else {
      CSC2CSR();
    }
  }
}

void Graph::CreateCSC() {
  if (csc_ != nullptr) return;
  if (coo_ != nullptr) {
    if (is_subgraph_) {
      val_col_ids_ = std::get<0>(torch::_unique(coo_->col));
      SetCSR(GraphCOO2DCSC(coo_, val_col_ids_.value(), true));
    } else {
      SetCSR(GraphCOO2CSC(coo_, num_cols_, true));
    }
  } else {
    if (is_subgraph_) {
      CSR2DCSC();
    } else {
      CSR2CSC();
    }
  }
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

// axis = 0 for col; axis = 1 for csr.
c10::intrusive_ptr<Graph> Graph::Slicing(torch::Tensor n_ids, int64_t axis,
                                         int64_t on_format,
                                         int64_t output_format) {
  c10::intrusive_ptr<Graph> ret;
  if (axis == 0)
    ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
        new Graph(true, n_ids, row_ids_, n_ids.numel(), num_rows_)));
  else
    ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
        new Graph(true, col_ids_, n_ids, num_cols_, n_ids.numel())));

  if (on_format == _COO) {
    CreateCOO();
    std::shared_ptr<COO> coo_ptr;
    torch::Tensor select_index, out_data;
    auto global_ids = axis == 0 ? col_ids_ : row_ids_;
    if (global_ids.has_value()) {
      LOG(FATAL) << "Not implementation";
    } else {
      std::tie(coo_ptr, select_index) = COOColSlicing(coo_, n_ids, axis);
    }
    ret->SetCOO(coo_ptr);
    ret->SetNumEdges(coo_ptr->row.numel());
    if (data_.has_value()) {
      if (coo_->e_ids.has_value())
        out_data =
            data_.value().index({coo_->e_ids.value().index({select_index})});
      else
        out_data = data_.value().index({select_index});
      ret->SetData(out_data);
    }

  } else if (on_format == _CSC) {
    if (output_format == _CSR) {
      LOG(FATAL) << "Error in Slicing, Not implementation [on_format = CSC, "
                    "output_forat = CSR]";
    }
    CreateCSC();
    torch::Tensor select_index, out_data;
    std::shared_ptr<_TMP> tmp_ptr;
    bool with_coo = output_format & _COO;
    if (axis == 0) {
      // for col
      if (val_col_ids_.has_value())
        std::tie(tmp_ptr, select_index) =
            DCSCColSlicing(csc_, val_col_ids_.value(), n_ids, with_coo);
      else if (col_ids_.has_value())
        std::tie(tmp_ptr, select_index) =
            DCSCColSlicing(csc_, col_ids_.value(), n_ids, with_coo);
      else
        std::tie(tmp_ptr, select_index) = CSCColSlicing(csc_, n_ids, with_coo);

    } else {
      // for row
      if (row_ids_.has_value())
        LOG(FATAL) << "Not implementation";
      else {
        std::tie(tmp_ptr, select_index) = CSCRowSlicing(csc_, n_ids, with_coo);
        torch::cuda::synchronize();
      }

      if (val_col_ids_.has_value()) ret->SetValidCols(val_col_ids_.value());
    }

    if (output_format & _CSC)
      ret->SetCSC(std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{
          tmp_ptr->coo_in_indices, tmp_ptr->coo_in_indptr, torch::nullopt}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csc_->e_ids.has_value()) {
        out_data =
            data_.value().index({csc_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else if (on_format == _CSR) {
    if (output_format == _CSC) {
      LOG(FATAL) << "Error in Slicing, Not implementation [on_format = CSR, "
                    "output_forat = CSC]";
    }
    CreateCSR();
    torch::Tensor select_index, out_data;
    std::shared_ptr<_TMP> tmp_ptr;
    bool with_coo = output_format & _COO;
    if (axis == 0) {
      // for col
      if (col_ids_.has_value())
        LOG(FATAL) << "Not implementation";
      else
        std::tie(tmp_ptr, select_index) = CSCRowSlicing(csr_, n_ids, with_coo);

      if (val_row_ids_.has_value()) ret->SetValidRows(val_row_ids_.value());

    } else {
      if (val_row_ids_.has_value())
        std::tie(tmp_ptr, select_index) =
            DCSCColSlicing(csr_, val_row_ids_.value(), n_ids, with_coo);
      else if (row_ids_.has_value())
        std::tie(tmp_ptr, select_index) =
            DCSCColSlicing(csr_, row_ids_.value(), n_ids, with_coo);
      else
        std::tie(tmp_ptr, select_index) = CSCColSlicing(csr_, n_ids, with_coo);
    }

    if (output_format & _CSR)
      ret->SetCSR(std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{
          tmp_ptr->coo_in_indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csr_->e_ids.has_value()) {
        out_data =
            data_.value().index({csr_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }
  } else {
    LOG(FATAL) << "Not implementation";
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::Sampling(int64_t axis, int64_t fanout,
                                          bool replace, int64_t on_format,
                                          int64_t output_format) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<_TMP> tmp_ptr;
  bool with_coo = output_format & _COO;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids_, num_cols_, num_rows_)));

  if (axis == 0 && on_format == _CSC) {
    if (output_format == _CSR)
      LOG(FATAL) << "Error in Sampling, Not implementation [on_format = CSC, "
                    "output_forat = CSR]";

    CreateCSC();
    std::tie(tmp_ptr, select_index) =
        CSCColSampling(csc_, fanout, replace, with_coo);

    if (output_format & _CSC)
      ret->SetCSC(std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{
          tmp_ptr->coo_in_indices, tmp_ptr->coo_in_indptr, torch::nullopt}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csc_->e_ids.has_value()) {
        out_data =
            data_.value().index({csc_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else if (axis == 1 && on_format == _CSR) {
    if (output_format == _CSC)
      LOG(FATAL) << "Error in Sampling, Not implementation [on_format = CSR, "
                    "output_forat = CSC]";

    CreateCSR();
    std::tie(tmp_ptr, select_index) =
        CSCColSampling(csr_, fanout, replace, with_coo);
    if (output_format & _CSR)
      ret->SetCSR(std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{
          tmp_ptr->coo_in_indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csr_->e_ids.has_value()) {
        out_data =
            data_.value().index({csr_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else {
    LOG(FATAL) << "No implementation!";
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::SamplingProbs(int64_t axis,
                                               torch::Tensor edge_probs,
                                               int64_t fanout, bool replace,
                                               int64_t on_format,
                                               int64_t output_format) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<_TMP> tmp_ptr;
  bool with_coo = output_format & _COO;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids_, num_cols_, num_rows_)));
  if (axis == 0 && on_format == _CSC) {
    if (output_format == _CSR)
      LOG(FATAL) << "Error in Sampling, Not implementation [on_format = CSC, "
                    "output_forat = CSR]";

    CreateCSC();
    std::tie(tmp_ptr, select_index) =
        CSCColSamplingProbs(csc_, edge_probs, fanout, replace, with_coo);

    if (output_format & _CSC)
      ret->SetCSC(std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{
          tmp_ptr->coo_in_indices, tmp_ptr->coo_in_indptr, torch::nullopt}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csc_->e_ids.has_value()) {
        out_data =
            data_.value().index({csc_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else if (axis == 1 && on_format == _CSR) {
    if (output_format == _CSC)
      LOG(FATAL) << "Error in Sampling, Not implementation [on_format = CSR, "
                    "output_forat = CSC]";

    CreateCSR();
    std::tie(tmp_ptr, select_index) =
        CSCColSamplingProbs(csr_, edge_probs, fanout, replace, with_coo);
    if (output_format & _CSR)
      ret->SetCSR(std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{
          tmp_ptr->coo_in_indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csr_->e_ids.has_value()) {
        out_data =
            data_.value().index({csr_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else {
    LOG(FATAL) << "No implementation!";
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
  int64_t num_rows =
      (row_ids_.has_value()) ? row_ids_.value().numel() : num_rows_;
  SetCSR(GraphCOO2CSC(coo_, num_rows, false));
}

void Graph::CSC2DCSR() {
  SetCOO(GraphCSC2COO(csc_, true));
  val_row_ids_ = std::get<0>(torch::_unique(coo_->row));
  SetCSR(GraphCOO2DCSC(coo_, val_row_ids_.value(), false));
}

void Graph::CSR2CSC() {
  SetCOO(GraphCSC2COO(csr_, false));
  int64_t num_cols =
      (col_ids_.has_value()) ? col_ids_.value().numel() : num_cols_;
  SetCSC(GraphCOO2CSC(coo_, num_cols, true));
}

void Graph::CSR2DCSC() {
  SetCOO(GraphCSC2COO(csr_, false));
  val_col_ids_ = std::get<0>(torch::_unique(coo_->col));
  SetCSR(GraphCOO2DCSC(coo_, val_col_ids_.value(), true));
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

torch::Tensor Graph::Sum(int64_t axis, int64_t powk, int64_t on_format) {
  auto in_data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto out_size = (axis == 0) ? num_cols_ : num_rows_;
  torch::Tensor out_data = torch::zeros(out_size, in_data.options());

  if (on_format == _COO) {
    CreateCOO();
    COOGraphSum(coo_, in_data, out_data, powk, axis);
  } else if (axis == 0 && on_format == _CSC) {
    CreateCSC();
    CSCGraphSum(csc_, val_col_ids_, in_data, out_data, powk);
  } else if (axis == 1 && on_format == _CSR) {
    CreateCSR();
    CSCGraphSum(csr_, val_row_ids_, in_data, out_data, powk);
  } else {
    LOG(FATAL) << "axis should be 0 or 1? on_format and axis do not match?";
  }
  return out_data;
}

c10::intrusive_ptr<Graph> Graph::Divide(torch::Tensor divisor, int64_t axis,
                                        int64_t on_format) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  auto in_data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor out_data = torch::zeros(num_edges_, in_data.options());
  if (on_format == _COO) {
    CreateCOO();
    COOGraphDiv(coo_, in_data, divisor, out_data, axis);
  } else if (axis == 0 && on_format == _CSC) {
    CreateCSC();
    CSCGraphDiv(csc_, val_col_ids_, in_data, divisor, out_data);
  } else if (axis == 1 && on_format == _CSR) {
    CreateCSR();
    CSCGraphDiv(csr_, val_row_ids_, in_data, divisor, out_data);
  }
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  ret->SetCOO(coo_);
  ret->SetNumEdges(num_edges_);
  ret->SetData(out_data);
  return ret;
}

c10::intrusive_ptr<Graph> Graph::Normalize(int64_t axis, int64_t on_format) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  auto in_data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor out_data = torch::zeros(num_edges_, in_data.options());
  if (on_format == _COO) {
    CreateCOO();
    if (axis == 0)
      COOGraphNormalize(coo_, in_data, out_data, num_cols_, axis);
    else
      COOGraphNormalize(coo_, in_data, out_data, num_rows_, axis);
  } else if (axis == 0 && on_format == _CSC) {
    CreateCSC();
    CSCGraphNormalize(csc_, val_col_ids_, in_data, out_data);
  } else if (axis == 1 && on_format == _CSR) {
    CreateCSR();
    CSCGraphNormalize(csr_, row_ids_, in_data, out_data);
  }
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  ret->SetCOO(coo_);
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
      SetCOO(GraphCSC2COO(csr_, false));
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
                  torch::Tensor out, int64_t lhs_target, int64_t rhs_target,
                  int64_t on_format) {
  const auto& bcast = CalcBcastOff(op, lhs, rhs);
  if (on_format == _COO) {
    CreateCOO();
    impl::SDDMMCOO(op, bcast, coo_, lhs, rhs, out, lhs_target, rhs_target);
  } else if (on_format == _CSR) {
    CreateCSR();
    lhs_target = lhs_target == 1 ? lhs_target : (2 - lhs_target);
    rhs_target = rhs_target == 1 ? rhs_target : (2 - rhs_target);
    impl::SDDMMCSC(op, bcast, csr_, val_row_ids_, lhs, rhs, out, lhs_target,
                   rhs_target);
  } else if (on_format == _CSC) {
    CreateCSC();
    impl::SDDMMCSC(op, bcast, csc_, val_row_ids_, lhs, rhs, out, lhs_target,
                   rhs_target);
  } else {
    LOG(FATAL) << "SDDMM only supports CSR and COO formats";
  }
}

}  // namespace gs
