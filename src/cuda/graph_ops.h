#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace gs {
namespace impl {

template<typename IdType>
  struct GraphKernelData {
    const IdType *in_ptr;
    const IdType *in_cols;
  };

std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids);

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

torch::Tensor MetapathRandomWalkFusedCUDA(torch::Tensor seeds,
  torch::Tensor metapath,
  torch::Tensor homo_indptr_tensor,
  torch::Tensor homo_indice_tensor, 
  torch::Tensor homo_indptr_offset_tensor,
  torch::Tensor homo_indices_offset_tensor);


}  // namespace impl
}  // namespace gs

#endif