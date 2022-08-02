#include "./graph.h"

namespace gs {

c10::intrusive_ptr<Graph> Graph::ColumnwiseSlicing(torch::Tensor column_ids) {
  // auto ret = c10::intrusive_ptr<Graph>(new Graph(true));
  // ret.SetCSC(CSCColumnwiseSlicing(csc_, column_ids));
  return c10::intrusive_ptr<Graph>();
}

}  // namespace gs