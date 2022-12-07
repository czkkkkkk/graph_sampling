#ifndef GS_CUDA_FUSION_SLICING_SAMPLING_H_
#define GS_CUDA_FUSION_SLICING_SAMPLING_H_
#include <torch/torch.h>

namespace gs {
namespace impl {
namespace fusion {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FusedCSCColSlicingSamplingCUDA(torch::Tensor indptr, torch::Tensor indices,
                               torch::Tensor column_ids, int64_t fanout,
                               bool replace);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FusedCSCColSlicingSamplingOneKeepDimCUDA(torch::Tensor indptr,
                                         torch::Tensor indices,
                                         torch::Tensor column_ids);
}  // namespace fusion
}  // namespace impl
}  // namespace gs

#endif