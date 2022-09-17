#include "./graph.h"

#include <sstream>

#include "./graph_ops.h"

namespace gs {

void Graph::LoadCSC(torch::Tensor indptr, torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->indptr = indptr;
  csc_->indices = indices;
  num_nodes_ = indptr.size(0) - 1;
  LOG(INFO) << "Loaded CSC with " << indptr.size(0) - 1 << " nodes and "
            << indices.size(0) << " edges";
}

void Graph::LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr,
                              torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  col_ids_ = torch::make_optional(column_ids);
  csc_->indptr = indptr;
  csc_->indices = indices;
  num_nodes_ = std::max(column_ids.max().item<int64_t>(),
                        indices.max().item<int64_t>()) +
               1;
}

std::shared_ptr<CSC> Graph::GetCSC() { return csc_; }

std::shared_ptr<CSR> Graph::GetCSR() { return csr_; }

std::shared_ptr<COO> Graph::GetCOO() { return coo_; }

torch::Tensor Graph::GetData() {
  if (data_.has_value()) {
    return data_.value();
  } else {
    torch::Tensor indices;
    if (csc_ != nullptr) {
      indices = csc_->indices;
    } else if (csr_ != nullptr) {
      indices = csr_->indices;
    } else {
      LOG(FATAL) << "Error in GetData: no CSC nor CSR";
    }
    return torch::ones(indices.numel(),
                       torch::dtype(torch::kFloat32).device(torch::kCUDA));
  }
}

int64_t Graph::GetNumNodes() { return num_nodes_; }

void Graph::SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }

void Graph::SetCSR(std::shared_ptr<CSR> csr) { csr_ = csr; }

void Graph::SetCOO(std::shared_ptr<COO> coo) { coo_ = coo; }

void Graph::SetData(torch::Tensor data) { data_ = torch::make_optional(data); }

c10::intrusive_ptr<Graph> Graph::ColumnwiseSlicing(torch::Tensor column_index) {
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_index})
                              : column_index;
  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(true, col_ids, row_ids_, num_nodes_)));
  ret->SetCSC(CSCColumnwiseSlicing(csc_, column_index));
  if (data_.has_value()) {
    ret->SetData(data_.value());
  }
  return ret;
}

// @todo Fix for new storage format (index and duplicated rows)
c10::intrusive_ptr<Graph> Graph::RowwiseSlicing(torch::Tensor row_index) {
  torch::Tensor row_ids =
      (row_ids_.has_value()) ? row_ids_.value().index({row_index}) : row_index;
  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(true, col_ids_, row_ids, num_nodes_)));
  if (csr_ != nullptr) {
    ret->SetCSR(CSRRowwiseSlicing(csr_, row_index));
  } else if (csc_ != nullptr) {
    ret->SetCSC(CSCRowwiseSlicing(csc_, row_index));
  } else {
    LOG(FATAL) << "Error in RowwiseSlicing: no CSC nor CSR";
  }
  if (data_.has_value()) {
    ret->SetData(data_.value());
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::ColumnwiseSampling(int64_t fanout,
                                                    bool replace) {
  torch::Tensor sampled_row_ids, unique_sorted_indices;
  auto csc_ptr = CSCColumnwiseSampling(csc_, fanout, replace);
  unique_sorted_indices = std::get<0>(torch::_unique(csc_ptr->indices));
  sampled_row_ids = (row_ids_.has_value())
                        ? row_ids_.value().index({unique_sorted_indices})
                        : unique_sorted_indices;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, sampled_row_ids, num_nodes_)));
  ret->SetCSC(csc_ptr);
  if (data_.has_value()) {
    ret->SetData(data_.value());
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::ColumnwiseFusedSlicingAndSampling(
    torch::Tensor column_index, int64_t fanout, bool replace) {
  torch::Tensor sampled_row_ids, unique_sorted_indices;
  auto csc_ptr =
      CSCColumnwiseFusedSlicingAndSampling(csc_, column_index, fanout, replace);
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_index})
                              : column_index;
  unique_sorted_indices = std::get<0>(torch::_unique(csc_ptr->indices));
  sampled_row_ids = (row_ids_.has_value())
                        ? row_ids_.value().index({unique_sorted_indices})
                        : unique_sorted_indices;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids, sampled_row_ids, num_nodes_)));
  ret->SetCSC(csc_ptr);
  if (data_.has_value()) {
    ret->SetData(data_.value());
  }
  return ret;
}

void Graph::CSC2CSR() {
  SetCOO(GraphCSC2COO(csc_));
  int64_t num_rows =
      (row_ids_.has_value()) ? row_ids_.value().numel() : num_nodes_;
  SetCSR(GraphCOO2CSR(coo_, num_rows));
}

void Graph::CSR2CSC() {
  SetCOO(GraphCSR2COO(csr_));
  int64_t num_cols =
      (col_ids_.has_value()) ? col_ids_.value().numel() : num_nodes_;
  SetCSC(GraphCOO2CSC(coo_, num_cols));
}

std::pair<torch::Tensor, torch::Tensor> Graph::PrepareDataForCompute(
    int64_t axis) {
  assertm(axis == 0 || axis == 1, "axis should be 0 or 1");
  torch::Tensor indptr, e_ids;
  if (axis == 0) {
    if (csc_ == nullptr) {
      CSR2CSC();
    }
    indptr = csc_->indptr;
    e_ids = (csc_->e_ids.has_value()) ? csc_->e_ids.value()
                                      : torch::arange(0, csc_->indices.numel(),
                                                      csc_->indices.options());
  } else {
    if (csr_ == nullptr) {
      CSC2CSR();
    }
    indptr = csr_->indptr;
    e_ids = (csr_->e_ids.has_value()) ? csr_->e_ids.value()
                                      : torch::arange(0, csr_->indices.numel(),
                                                      csr_->indices.options());
  }
  return {indptr, e_ids};
}

torch::Tensor Graph::Sum(int64_t axis) {
  torch::Tensor indptr, out_data, e_ids;
  std::tie(indptr, e_ids) = PrepareDataForCompute(axis);
  out_data = GraphSum(indptr, e_ids, GetData());
  return out_data;
}

torch::Tensor Graph::L2Norm(int64_t axis) {
  torch::Tensor indptr, out_data, e_ids;
  std::tie(indptr, e_ids) = PrepareDataForCompute(axis);
  out_data = GraphL2Norm(indptr, e_ids, GetData());
  return out_data;
}

c10::intrusive_ptr<Graph> Graph::Divide(torch::Tensor divisor, int64_t axis) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_nodes_)));
  torch::Tensor indptr, out_data, e_ids;
  std::tie(indptr, e_ids) = PrepareDataForCompute(axis);
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  out_data = GraphDiv(indptr, e_ids, GetData(), divisor);
  ret->SetData(out_data);
  return ret;
}

c10::intrusive_ptr<Graph> Graph::Normalize(int64_t axis) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_nodes_)));
  torch::Tensor indptr, out_data, e_ids;
  std::tie(indptr, e_ids) = PrepareDataForCompute(axis);
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  out_data = GraphNormalize(indptr, e_ids, GetData());
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::string> Graph::Relabel() {
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
    std::tie(frontier, relabeled_indices, relabeled_indptr) = GraphRelabel(col_ids, csc_->indptr, row_indices);
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
    std::tie(frontier, relabeled_indices, relabeled_indptr) = GraphRelabel(row_ids, csr_->indptr, col_indices);
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
  if (csc_ != nullptr) {
    if (col_ids_.has_value()) {
      return {col_ids_.value(), csc_->indptr, csc_->indices};
    } else {
      return {torch::Tensor(), csc_->indptr, csc_->indices};
    }
  } else if (csr_ != nullptr) {
    if (row_ids_.has_value()) {
      return {col_ids_.value(), csc_->indptr, csc_->indices};
    } else {
      return {torch::Tensor(), csc_->indptr, csc_->indices};
    }
  } else {
    LOG(FATAL) << "Error in MetaData: no CSC nor CSR.";
    return {coo_->row, coo_->col};
  }
}

}  // namespace gs
