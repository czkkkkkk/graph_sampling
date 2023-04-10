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

std::tuple<torch::Tensor, torch::Tensor> BatchListSamplingProbs(
    torch::Tensor probs, int64_t num_picks, bool replace, torch::Tensor range) {
  return impl::BatchListSamplingProbsCUDA(probs, num_picks, replace, range);
}

torch::Tensor IndexSearch(torch::Tensor origin_data, torch::Tensor keys) {
  torch::Tensor key_buffer, value_buffer;

  std::tie(key_buffer, value_buffer) =
      impl::IndexHashMapInsertCUDA(origin_data);
  torch::Tensor result =
      impl::IndexHashMapSearchCUDA(key_buffer, value_buffer, keys);
  return result;
}

std::vector<torch::Tensor> SplitByOffset(torch::Tensor data,
                                         torch::Tensor offset) {
  int64_t numel = offset.numel();
  torch::Tensor size_tensor =
      offset.slice(0, 1, numel) - offset.slice(0, 0, numel - 1);

  size_tensor = size_tensor.to(torch::kCPU);
  if(data.scalar_type()==torch::kInt64){
   // std::cout<<__FILE__<<":"<<__LINE__<<std::endl;
    auto data_ptr = size_tensor.data_ptr<int64_t>();
    std::vector<int64_t> split(data_ptr, data_ptr + size_tensor.numel());
    return torch::split_with_sizes(data, split);
  }
  else{
   // std::cout<<__FILE__<<":"<<__LINE__<<std::endl;
    auto data_ptr = size_tensor.data_ptr<int64_t>();
    std::vector<int64_t> split(data_ptr, data_ptr + size_tensor.numel());
    // std::vector<int64_t> split64(split.begin(), split.end()); // 将 std::vector<int32_t> 转换为 std::vector<int64_t>
    at::IntArrayRef split_sizes(split.data(), split.size()); // 使用新的 split64 变量
    return torch::split_with_sizes(data, split_sizes);
  }

}

}  // namespace gs
