#include "./graph.h"

namespace gs {

c10::intrusive_ptr<Graph> Graph::ColumnwiseSlicing(torch::Tensor column_ids) {
  auto tensor = this->test_trace_data_ + column_ids;
  auto ret_graph = Graph(tensor);
  return c10::make_intrusive<Graph>(ret_graph);
}

torch::Tensor Graph::Get() { return this->test_trace_data_; }

}  // namespace gs