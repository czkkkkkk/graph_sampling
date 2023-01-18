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

torch::Tensor IndexSearch(torch::Tensor origin_data, torch::Tensor keys) {
  torch::Tensor key_buffer, value_buffer;

  std::tie(key_buffer, value_buffer) =
      impl::IndexHashMapInsertCUDA(origin_data);
  torch::Tensor result =
      impl::IndexHashMapSearchCUDA(key_buffer, value_buffer, keys);
  return result;
}

}  // namespace gs
