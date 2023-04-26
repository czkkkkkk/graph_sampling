#ifndef GS_GRAPH_OPS_H_
#define GS_GRAPH_OPS_H_

#include <torch/torch.h>

#include "graph_storage.h"

namespace gs {
std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc, bool CSC2COO);

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_items,
                                  bool COO2CSC);

torch::Tensor FusedRandomWalk(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length);
torch::Tensor FusedNode2Vec(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                            int64_t walk_length, double p, double q);

}  // namespace gs

#endif