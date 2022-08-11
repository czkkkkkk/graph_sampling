#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gs {
namespace impl {


std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSlicingCUDA(torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids);

}
}

#endif