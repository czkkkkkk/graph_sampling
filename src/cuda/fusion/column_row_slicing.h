#ifndef GS_CUDA_FUSION_COLUMN_ROW_SLICING_H_
#define GS_CUDA_FUSION_COLUMN_ROW_SLICING_H_

#include <torch/torch.h>

namespace gs {
namespace impl {
namespace fusion {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCColRowSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor seeds);

}

}  // namespace impl
}  // namespace gs

#endif