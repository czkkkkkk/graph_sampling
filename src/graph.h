#ifndef GS_GRAPH_H_
#define GS_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph_storage.h"

namespace gs {

class Graph : public torch::CustomClassHolder {
 public:
  Graph(bool is_subgraph) { is_subgraph_ = is_subgraph; }
  c10::intrusive_ptr<Graph> ColumnwiseSlicing(torch::Tensor column_ids);

 private:
  bool is_subgraph_;
  std::shared_ptr<CSC> csc_;
};

}  // namespace gs

#endif