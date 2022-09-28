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

/**
 * @brief Returns the set of all nodes of the graph. Nodes in return tensor are
 * sorted in the order of their first occurrence in {col_ids, indices}.
 * For example,
 *    if graph.csc_.col_ids = [0, 2, 4, 2] and graph.csc_.indices = [4, 2, 1],
 *    graph.AllIndices() will be [0, 2, 4, 1]
 *
 * @return torch::Tensor
 */
torch::Tensor Graph::AllIndices(bool unique) {
  torch::Tensor col_ids, row_ids, cat;
  if (csc_ != nullptr) {
    col_ids = (col_ids_.has_value()) ? col_ids_.value()
                                     : torch::arange(csc_->indptr.numel() - 1,
                                                     csc_->indptr.options());
    row_ids = (row_ids_.has_value()) ? row_ids_.value() : csc_->indices;
    cat = torch::cat({col_ids, row_ids});
  } else if (csr_ != nullptr) {
    col_ids = (col_ids_.has_value()) ? col_ids_.value() : csr_->indices;
    row_ids = (row_ids_.has_value()) ? row_ids_.value()
                                     : torch::arange(csr_->indptr.numel() - 1,
                                                     csr_->indptr.options());
    cat = torch::cat({col_ids, row_ids});
  } else {
    LOG(FATAL) << "Error in RowIndices: no CSC nor CSR";
    cat = torch::Tensor();
  }
  return (unique) ? TensorUnique(cat) : cat;
}

/**
 * @todo Need relabel or not?
 * @todo Fix for CSC and CSR.
 * @todo Fix for new storage format.
 *
 * @brief Do relabel operation on graph.col_ids and graph.indices;
 * It will return {all_indices, new_csc_indptr, new_csc_indices}.
 * Specifically, all_indices = graph.AllIndices(); new_csc_indptr is the
 * csc_indptr of the relabeled graph; new_csc_indices is the csc_indices of the
 * relabeled graph.
 * For example,
 *    if graph.csc_.col_ids = [0, 2, 4, 2], graph.csc_.indptr = [0, 0, 1, 1, 3]
 *    and graph.csc_.indices = [4, 2, 1],
 *    graph.relabel will return {[0, 2, 4, 1], [0, 0, 1, 1, 3], [2, 1, 3]}
 *
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::string>
Graph::Relabel() {
  torch::Tensor frontier, relabeled_indices, relabeled_indptr;
  if (csc_ != nullptr) {
    torch::Tensor col_ids, row_indices;
    if (col_ids_.has_value()) {
      col_ids = col_ids_.value();
    } else {
      col_ids = torch::arange(csc_->indptr.numel() - 1, csc_->indptr.options());
    }
    if (row_ids_.has_value()) {
      row_indices = row_ids_.value().index({csc_->indices});
    } else {
      row_indices = csc_->indices;
    }
    std::tie(frontier, relabeled_indices, relabeled_indptr) =
        GraphRelabel(col_ids, csc_->indptr, row_indices);
    return {frontier, relabeled_indices, relabeled_indptr, "csc"};
  } else if (csr_ != nullptr) {
    torch::Tensor row_ids, col_indices;
    if (row_ids_.has_value()) {
      row_ids = row_ids_.value();
    } else {
      row_ids = torch::arange(csr_->indptr.numel() - 1, csr_->indptr.options());
    }
    if (col_ids_.has_value()) {
      col_indices = col_ids_.value().index({csr_->indices});
    } else {
      col_indices = csr_->indices;
    }
    std::tie(frontier, relabeled_indices, relabeled_indptr) =
        GraphRelabel(row_ids, csr_->indptr, col_indices);
    return {frontier, relabeled_indices, relabeled_indptr, "csr"};
  } else {
    LOG(FATAL) << "Error in relabel: no CSC nor CSR.";
    return {};
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
