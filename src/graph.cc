#include "./graph.h"

#include <sstream>

#include "./graph_ops.h"

namespace gs {

void Graph::LoadCSC(torch::Tensor indptr, torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->indptr = indptr;
  csc_->indices = indices;
}

void Graph::SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }

c10::intrusive_ptr<Graph> Graph::ColumnwiseSlicing(torch::Tensor column_ids) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(true)));
  ret->SetCSC(CSCColumnwiseSlicing(csc_, column_ids));
  return ret;
}

torch::Tensor Graph::RowIndices() { return torch::Tensor(); }

void Graph::Print() const {
  std::stringstream ss;
  ss << "# Nodes: " << csc_->indptr.size(0) - 1
     << " # Edges: " << csc_->indices.size(0) << "\n";
  ss << "CSC indptr: " << csc_->indptr << "\n";
  ss << "CSC indices: " << csc_->indices << "\n";
  std::cout << ss.str();
}

torch::Tensor Graph::Get() { return this->test_trace_data_; }

}  // namespace gs