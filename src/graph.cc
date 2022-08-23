#include "./graph.h"
#include <sstream>
#include "./graph_ops.h"

namespace gs {

void Graph::LoadCSC(torch::Tensor indptr, torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->indptr = indptr;
  csc_->indices = indices;
}

void Graph::LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr, torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->col_ids = column_ids;
  csc_->indptr = indptr;
  csc_->indices = indices;
}

std::shared_ptr<CSC> Graph::GetCSC() { return csc_; }

void Graph::SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }

c10::intrusive_ptr<Graph> Graph::ColumnwiseSlicing(torch::Tensor column_ids) {
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(true)));
  ret->SetCSC(CSCColumnwiseSlicing(csc_, column_ids));
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

torch::Tensor Graph::RowIndices() { return torch::Tensor(); }

torch::Tensor Graph::AllIndices() {
  if(is_subgraph_){
  torch::Tensor cat = torch::cat({csc_->col_ids, csc_->indices});
  return TensorUnique(cat);
  }
  else{
  int64_t size = csc_->indptr.numel();
  //torch::Tensor nodeids = torch::arange(size).to(torch::kCUDA);
  torch::Tensor cat = torch::cat({ csc_->indptr.slice(0, 0, size-1),csc_->indices});
  return TensorUnique(cat);
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Graph::Relabel(){
  return GraphRelabel(csc_->col_ids, csc_->indptr, csc_->indices);
}

void Graph::Print() const {
  std::stringstream ss;
  if(is_subgraph_){ 
     ss << "col ids: "    << csc_->col_ids << "\n";
  }
  ss << "# Nodes: " << csc_->indptr.size(0) - 1
     << " # Edges: " << csc_->indices.size(0) << "\n";
  ss << "CSC indptr: "
     << "\n"
     << csc_->indptr << "\n";
  ss << "CSC indices: "
     << "\n"
     << csc_->indices << "\n";
  std::cout << ss.str();
}

std::vector<torch::Tensor> Graph::MetaData() {
  return {csc_->col_ids, csc_->indptr, csc_->indices};
}

}  // namespace gs

