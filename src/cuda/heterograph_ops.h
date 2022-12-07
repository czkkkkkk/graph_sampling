#ifndef GS_CUDA_HETEROGRAPH_OPS_H_
#define GS_CUDA_HETEROGRAPH_OPS_H_

#include <torch/torch.h>

namespace gs {
namespace impl {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseSamplingOneKeepDimCUDA(torch::Tensor indptr, torch::Tensor indices,
                                    torch::Tensor column_ids);

}  // namespace impl
}  // namespace gs

#endif