#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <torch/torch.h>

namespace gs {
namespace impl {

std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids);

std::pair<torch::Tensor, torch::Tensor> CSCRowwiseSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_ids);

torch::Tensor TensorUniqueCUDA(torch::Tensor input);

std::tuple<torch::Tensor, torch::Tensor> RelabelCUDA(torch::Tensor col_ids,
                                                     torch::Tensor indices);

std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSamplingCUDA(
    torch::Tensor indptr, torch::Tensor indices, int64_t fanout, bool replace);

std::pair<torch::Tensor, torch::Tensor>
CSCColumnwiseFusedSlicingAndSamplingCUDA(torch::Tensor indptr,
                                         torch::Tensor indices,
                                         torch::Tensor column_ids,
                                         int64_t fanout, bool replace);

torch::Tensor RandomWalkFusedCUDA(
    torch::Tensor seeds, int64_t walklength,
    int64_t* all_indices,
    int64_t* all_indptr);

}  // namespace impl
}  // namespace gs

#endif