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

std::tuple<torch::Tensor, torch::Tensor> ListSamplingProbsCUDA(
    torch::Tensor data, torch::Tensor probs, int64_t num_picks, bool replace);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BatchListSamplingProbsCUDA(torch::Tensor data, torch::Tensor probs,
                           int64_t num_picks, bool replace,
                           torch::Tensor range);

torch::Tensor TensorUniqueCUDA(torch::Tensor input);

// RelabelCUDA leverages vector<Tensor> mapping_tensor to create the hashmap
// which stores the mapping. Then, it will do relabel operation for tensor in
// data_requiring_relabel with the hashmap.
// It return {unique_tensor, {tensor1_after_relabeled,
// tensor2_after_relabeled, ...}}.
std::tuple<torch::Tensor, std::vector<torch::Tensor>> RelabelCUDA(
    const std::vector<torch::Tensor> &mapping_tensor,
    const std::vector<torch::Tensor> &data_requiring_relabel);

torch::Tensor IndexSelectCPUFromGPU(torch::Tensor array, torch::Tensor index);

std::tuple<torch::Tensor, torch::Tensor> IndexHashMapInsertCUDA(
    torch::Tensor keys);

torch::Tensor IndexHashMapSearchCUDA(torch::Tensor key_buffer,
                                     torch::Tensor value_buffer,
                                     torch::Tensor keys);

std::vector<torch::Tensor> SplitIndptrBySizeCUDA(torch::Tensor indptr,
                                                 int64_t size);
std::vector<torch::Tensor> SplitIndptrByOffsetCUDA(torch::Tensor indptr,
                                                   torch::Tensor offsets);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchConcatCUDA(
    const std::vector<torch::Tensor> &data_tensors,
    const std::vector<torch::Tensor> &offset_tensors);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUniqueByKeyCUDA(
    torch::Tensor data_tensor, torch::Tensor data_ptr, torch::Tensor data_key);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUniqueCUDA(
    torch::Tensor data_tensor, torch::Tensor data_ptr);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BatchRelabelByKeyCUDA(torch::Tensor data_tensor, torch::Tensor data_ptr,
                      torch::Tensor data_key);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BatchRelabelCUDA(torch::Tensor data_tensor, torch::Tensor data_ptr);
}  // namespace impl
}  // namespace gs

#endif