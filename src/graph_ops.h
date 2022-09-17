#ifndef GS_GRAPH_OPS_H_
#define GS_GRAPH_OPS_H_

#include <torch/torch.h>

#include "graph_storage.h"

namespace gs {

std::shared_ptr<CSC> CSCColumnwiseSlicing(std::shared_ptr<CSC> csc,
                                          torch::Tensor column_ids);

std::shared_ptr<CSC> CSCRowwiseSlicing(std::shared_ptr<CSC> csc,
                                       torch::Tensor row_ids);

std::shared_ptr<CSR> CSRRowwiseSlicing(std::shared_ptr<CSR> csr,
                                       torch::Tensor row_ids);

std::shared_ptr<CSC> CSCColumnwiseSampling(std::shared_ptr<CSC> csc,
                                           int64_t fanout, bool replace);

std::shared_ptr<CSC> CSCColumnwiseFusedSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, int64_t fanout,
    bool replace);

torch::Tensor TensorUnique(torch::Tensor node_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GraphRelabel(
    torch::Tensor col_ids, torch::Tensor indptr, torch::Tensor indices);

torch::Tensor GraphSum(torch::Tensor indptr, torch::Tensor e_ids,
                       torch::Tensor data);

torch::Tensor GraphDiv(torch::Tensor indptr, torch::Tensor e_ids,
                       torch::Tensor data, torch::Tensor divisor);

torch::Tensor GraphL2Norm(torch::Tensor indptr, torch::Tensor e_ids,
                          torch::Tensor data);

torch::Tensor GraphNormalize(torch::Tensor indptr, torch::Tensor e_ids,
                             torch::Tensor data);

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc);

std::shared_ptr<COO> GraphCSR2COO(std::shared_ptr<CSR> csr);

std::shared_ptr<CSR> GraphCOO2CSR(std::shared_ptr<COO> coo, int64_t num_rows);

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_cols);

}  // namespace gs

#endif