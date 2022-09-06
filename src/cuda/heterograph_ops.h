#ifndef GS_CUDA_HETEROGRAPH_OPS_H_
#define GS_CUDA_HETEROGRAPH_OPS_H_

#include <thrust/device_vector.h>
#include <torch/torch.h>

namespace gs {
namespace impl {

std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSamplingOneKeepDimCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids);

torch::Tensor MetapathRandomWalkFusedCUDA(
    torch::Tensor seeds, torch::Tensor metapath,
    thrust::device_vector<int64_t *> &all_indices,
    thrust::device_vector<int64_t *> &all_indptr);

}  // namespace impl
}  // namespace gs

#endif