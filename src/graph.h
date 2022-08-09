#ifndef GS_GRAPH_H_
#define GS_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph_storage.h"

namespace gs {

class Graph : public torch::CustomClassHolder {
 public:
  Graph(torch::Tensor test_data) { test_trace_data_ = test_data; }
  c10::intrusive_ptr<Graph> ColumnwiseSlicing(torch::Tensor column_ids);
  torch::Tensor Get();

 private:
  bool is_subgraph_;
  std::shared_ptr<CSC> csc_;
  torch::Tensor test_trace_data_;
};

}  // namespace gs

#endif