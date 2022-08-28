#ifndef GS_CUDA_TENSOR_OPS_H_
#define GS_CUDA_TENSOR_OPS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace gs {
namespace impl {

std::tuple<torch::Tensor, torch::Tensor> ListSamplingCUDA(torch::Tensor data,
                                                      int64_t num_picks,
                                                      bool replace);

}  // namespace impl
}  // namespace gs

#endif