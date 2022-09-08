#include "./graph.h"

#include <sstream>

#include "./graph_ops.h"

namespace gs {

void Graph::LoadCSC(torch::Tensor indptr, torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->indptr = indptr;
  csc_->indices = indices;
  LOG(INFO) << "Loaded CSC with " << indptr.size(0) - 1 << " nodes and "
            << indices.size(0) << " edges";
}

void Graph::LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr,
                              torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->col_ids = column_ids;
  csc_->indptr = indptr;
  csc_->indices = indices;
}

std::shared_ptr<CSC> Graph::GetCSC() { return csc_; }

std::shared_ptr<CSR> Graph::GetCSR() { return csr_; }

std::shared_ptr<COO> Graph::GetCOO() { return coo_; }

void Graph::SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }

void Graph::SetCSR(std::shared_ptr<CSR> csr) { csr_ = csr; }

void Graph::SetCOO(std::shared_ptr<COO> coo) { coo_ = coo; }

c10::intrusive_ptr<Graph> Graph::ColumnwiseSlicing(torch::Tensor column_ids) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(true)));
  ret->SetCSC(CSCColumnwiseSlicing(csc_, column_ids));
  return ret;
}

c10::intrusive_ptr<Graph> Graph::RowwiseSlicing(torch::Tensor row_ids) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(true)));
  ret->SetCSC(CSCRowwiseSlicing(csc_, row_ids));
  return ret;
}

c10::intrusive_ptr<Graph> Graph::ColumnwiseSampling(int64_t fanout,
                                                    bool replace) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(true)));
  ret->SetCSC(CSCColumnwiseSampling(csc_, fanout, replace));
  return ret;
}

c10::intrusive_ptr<Graph> Graph::ColumnwiseFusedSlicingAndSampling(
    torch::Tensor column_ids, int64_t fanout, bool replace) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(true)));
  ret->SetCSC(
      CSCColumnwiseFusedSlicingAndSampling(csc_, column_ids, fanout, replace));
  return ret;
}

void Graph::CSC2CSR() {
  SetCOO(GraphCSC2COO(csc_));
  SetCSR(GraphCOO2CSR(coo_));
}

// c10::intrusive_ptr<Graph> Graph::Normalize(int axis) {
//   auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new
//   Graph(true))); assertm(axis == 0 || axis == 1, "axis should be 0 or 1"); if
//   (axis == 0) {
//     ret->SetCSC(GraphNormalize(csc_));
//   } else {
//     LOG(FATAL) << "Not implemented warning";
//   }
//   return ret;
// }

torch::Tensor Graph::RowIndices() { return TensorUnique(csc_->indices); }

/**
 * @brief Returns the set of all nodes of the graph. Nodes in return tensor are
 * sorted in the order of their first occurrence in {col_ids, indices}.
 * For example,
 *    if graph.csc_.col_ids = [0, 2, 4, 2] and graph.csc_.indices = [4, 2, 1],
 *    graph.AllIndices() will be [0, 2, 4, 1]
 *
 * @return torch::Tensor
 */
torch::Tensor Graph::AllIndices() {
  if (is_subgraph_) {
    torch::Tensor cat = torch::cat({csc_->col_ids, csc_->indices});
    return TensorUnique(cat);
  } else {
    int64_t size = csc_->indptr.numel();
    // torch::Tensor nodeids = torch::arange(size).to(torch::kCUDA);
    torch::Tensor cat = torch::cat(
        {torch::arange(size - 1, csc_->indptr.options()), csc_->indices});
    return TensorUnique(cat);
  }
}

/**
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Graph::Relabel() {
  return GraphRelabel(csc_->col_ids, csc_->indptr, csc_->indices);
}

void Graph::Print() const {
  std::stringstream ss;
  if (is_subgraph_) {
    ss << "col ids: " << csc_->col_ids << "\n";
  }
  ss << "# Nodes: " << csc_->indptr.size(0) - 1
     << " # Edges: " << csc_->indices.size(0) << "\n";
  ss << "CSC indptr: "
     << "\n"
     << csc_->indptr << "\n";
  ss << "CSC indices: "
     << "\n"
     << csc_->indices << "\n";
  LOG(INFO) << ss.str();
}

std::vector<torch::Tensor> Graph::MetaData() {
  return {csc_->col_ids, csc_->indptr, csc_->indices};
}

}  // namespace gs
