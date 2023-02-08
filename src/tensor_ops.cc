#include "./tensor_ops.h"
#include "cuda/tensor_ops.h"

namespace gs {
std::tuple<torch::Tensor, torch::Tensor> ListSampling(torch::Tensor data,
                                                      int64_t num_picks,
                                                      bool replace) {
  return impl::ListSamplingCUDA(data, num_picks, replace);
}

std::tuple<torch::Tensor, torch::Tensor> ListSamplingProbs(torch::Tensor data,
                                                           torch::Tensor probs,
                                                           int64_t num_picks,
                                                           bool replace) {
  return impl::ListSamplingProbsCUDA(data, probs, num_picks, replace);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchListSamplingProbs(
    torch::Tensor data, torch::Tensor probs, int64_t num_picks, bool replace,
    torch::Tensor range) {
  return impl::BatchListSamplingProbsCUDA(data, probs, num_picks, replace,
                                          range);
}

torch::Tensor IndexSearch(torch::Tensor origin_data, torch::Tensor keys) {
  torch::Tensor key_buffer, value_buffer;

  std::tie(key_buffer, value_buffer) =
      impl::IndexHashMapInsertCUDA(origin_data);
  torch::Tensor result =
      impl::IndexHashMapSearchCUDA(key_buffer, value_buffer, keys);
  return result;
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
BatchRelabel(const std::vector<torch::Tensor> &batch_tensors,
             const std::vector<torch::Tensor> &segment_ptrs,
             int64_t num_batchs) {
  return impl::BatchRelabelCUDA(batch_tensors, segment_ptrs, num_batchs);
}

std::vector<torch::Tensor> SplitByOffset(torch::Tensor data,
                                         torch::Tensor offset) {
  int64_t numel = offset.numel();
  torch::Tensor size_tensor =
      offset.slice(0, 1, numel) - offset.slice(0, 0, numel - 1);

  size_tensor = size_tensor.to(torch::kCPU);

  auto data_ptr = size_tensor.data_ptr<int64_t>();
  std::vector<int64_t> split(data_ptr, data_ptr + size_tensor.numel());

  return torch::split_with_sizes(data, split);
}

}  // namespace gs
