#include "./tensor_ops.h"
#include "cuda/tensor_ops.h"

namespace gs {
std::tuple<torch::Tensor, torch::Tensor> ListSampling(torch::Tensor data,
                                                      int64_t num_picks,
                                                      bool replace) {
  return impl::ListSamplingCUDA(data, num_picks, replace);
}
}  // namespace gs
