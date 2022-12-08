#ifndef GS_GRAPH_OPS_H_
#define GS_GRAPH_OPS_H_

#include <torch/torch.h>

#include "graph_storage.h"

namespace gs {

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCColSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids);

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCRowSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids);

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCColSampling(
    std::shared_ptr<CSC> csc, int64_t fanout, bool replace);

std::pair<std::shared_ptr<CSC>, torch::Tensor>
FusedCSCColSlicingAndSampling(std::shared_ptr<CSC> csc,
                                     torch::Tensor node_ids, int64_t fanout,
                                     bool replace);

torch::Tensor TensorUnique(torch::Tensor node_ids);

std::tuple<torch::Tensor, std::vector<torch::Tensor>> BatchTensorRelabel(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> to_be_relabeled_tensors);

torch::Tensor GraphSum(std::shared_ptr<CSC> csc,
                       torch::optional<torch::Tensor> data, int64_t powk);

torch::Tensor GraphDiv(std::shared_ptr<CSC> csc,
                       torch::optional<torch::Tensor> data,
                       torch::Tensor divisor);

torch::Tensor GraphNormalize(std::shared_ptr<CSC> csc,
                             torch::optional<torch::Tensor> data);

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc, bool CSC2COO);

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_items,
                                  bool COO2CSC);

torch::Tensor FusedRandomWalk(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length);

}  // namespace gs

#endif