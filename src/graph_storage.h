#ifndef GS_GRAPH_STORAGE_H_
#define GS_GRAPH_STORAGE_H_

#include "torch/custom_class.h"

namespace gs {

struct CSC {
  int64_t n_nodes, n_edges;
  torch::Tensor indptr, indices;
  at::optional<torch::Tensor> src_indices, dst_indices;
  std::map<std::string, torch::Tensor> edge_data;
  at::optional<torch::Tensor> eid;
};

}  // namespace gs

#endif