#ifndef GS_CUDA_FUSION_FUSION_GRAPH_OPS_H_
#define GS_CUDA_FUSION_FUSION_GRAPH_OPS_H_

#include <torch/torch.h>
#include "./logging.h"

namespace gs {
namespace impl {
namespace fusion {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor seeds);

}

}  // namespace impl
}  // namespace gs

#endif