#ifndef GS_GRAPH_OPS_H_
#define GS_GRAPH_OPS_H_

#include <torch/torch.h>

#include "graph_storage.h"

namespace gs {

std::shared_ptr<CSC> CSCColumnwiseSlicing(std::shared_ptr<CSC> csc,
                                          torch::Tensor column_ids);

torch::Tensor TensorUnique(torch::Tensor node_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GraphRelabel(
    torch::Tensor col_ids, torch::Tensor indptr, torch::Tensor indices);

std::shared_ptr<CSC> CSCColumnwiseSampling(std::shared_ptr<CSC> csc,
                                           int64_t fanout, bool replace);

std::shared_ptr<CSC> CSCColumnwiseFusedSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, int64_t fanout,
    bool replace);

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc);

std::shared_ptr<CSR> GraphCOO2CSR(std::shared_ptr<COO> coo);

}  // namespace gs

#endif