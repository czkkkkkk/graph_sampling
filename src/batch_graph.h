#ifndef GS_BATCH_GRAPH_H_
#define GS_BATCH_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include "graph.h"
#include "graph_storage.h"

namespace gs {

class BatchGraph : public Graph {
 public:
  // init graph
  // python code will make sure that all inputs are legitimate
  using Graph::Graph;
  // graph operation
  std::tuple<c10::intrusive_ptr<BatchGraph>, torch::Tensor> BatchColSlicing(
      torch::Tensor seeds, torch::Tensor batch_ptr, int64_t axis,
      int64_t on_format, int64_t output_format, bool encoding);

  std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
             std::vector<torch::Tensor>, std::vector<torch::Tensor>,
             std::string>
  GraphRelabel(torch::Tensor col_seeds, torch::Tensor row_ids);

  void SetEdgeBptr(torch::Tensor bptr) { edge_bptr_ = bptr; }

 private:
  torch::Tensor col_bptr_, edge_bptr_;
};

}  // namespace gs

#endif
