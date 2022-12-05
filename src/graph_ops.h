#ifndef GS_GRAPH_OPS_H_
#define GS_GRAPH_OPS_H_

#include <torch/torch.h>

#include "graph_storage.h"

namespace gs {

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCFusionSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor seeds);

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCColumnwiseSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids);

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCRowwiseSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor row_ids);

std::pair<std::shared_ptr<CSR>, torch::Tensor> CSRRowwiseSlicing(
    std::shared_ptr<CSR> csr, torch::Tensor row_ids);

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCColumnwiseSampling(
    std::shared_ptr<CSC> csc, int64_t fanout, bool replace);

std::pair<std::shared_ptr<CSC>, torch::Tensor>
CSCColumnwiseFusedSlicingAndSampling(std::shared_ptr<CSC> csc,
                                     torch::Tensor column_ids, int64_t fanout,
                                     bool replace);

torch::Tensor TensorUnique(torch::Tensor node_ids);

std::tuple<torch::Tensor, std::vector<torch::Tensor>> BatchTensorRelabel(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> to_be_relabeled_tensors);

torch::Tensor GraphSum(std::shared_ptr<CSC> csc,
                       torch::optional<torch::Tensor> data, int64_t powk);

torch::Tensor GraphSum(std::shared_ptr<CSR> csr,
                       torch::optional<torch::Tensor> data, int64_t powk);

torch::Tensor GraphDiv(std::shared_ptr<CSC> csc,
                       torch::optional<torch::Tensor> data,
                       torch::Tensor divisor);

torch::Tensor GraphDiv(std::shared_ptr<CSR> csr,
                       torch::optional<torch::Tensor> data,
                       torch::Tensor divisor);

torch::Tensor GraphNormalize(std::shared_ptr<CSC> csc,
                             torch::optional<torch::Tensor> data);

torch::Tensor GraphNormalize(std::shared_ptr<CSR> csr,
                             torch::optional<torch::Tensor> data);

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc);

std::shared_ptr<COO> GraphCSR2COO(std::shared_ptr<CSR> csr);

std::shared_ptr<CSR> GraphCOO2CSR(std::shared_ptr<COO> coo, int64_t num_rows);

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_cols);

torch::Tensor RandomWalkFused(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length);

}  // namespace gs

#endif