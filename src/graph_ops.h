#ifndef GS_GRAPH_OPS_H_
#define GS_GRAPH_OPS_H_

#include <torch/torch.h>

#include "graph_storage.h"

namespace gs {

std::shared_ptr<CSC> CSCColumnwiseSlicing(std::shared_ptr<CSC> csc,
                                          torch::Tensor column_ids);

std::shared_ptr<CSC> CSCColumnwiseSampling(std::shared_ptr<CSC> csc,
                                           int64_t fanout, bool replace);

std::shared_ptr<CSC> CSCColumnwiseFusedSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, int64_t fanout,
    bool replace);

}  // namespace gs

#endif