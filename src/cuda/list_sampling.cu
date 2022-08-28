#include <curand_kernel.h>
#include <tuple>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"

namespace gs {
namespace impl {

template <typename IdType>
__global__ void _ListSamplingIndexKernel(const uint64_t rand_seed,
                                         const IdType *const in_data,
                                         int64_t *const out_index,
                                         const int num_items,
                                         const int num_picks) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int tot = blockDim.x * gridDim.x;

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  for (int64_t idx = num_picks + tid; idx < num_items; idx += tot) {
    int64_t num = curand(&rng) % (idx + 1);
    if (num < num_picks) {
      AtomicMax(out_index + num, idx);
    }
  }
}

/**
 * @brief ListSampling, using A-Res sampling for replace = False and uniform
 * sampling for replace = True. It will return (selected_data, selected_index)
 *
 * @tparam IdType
 * @param data
 * @param num_picks
 * @param replace
 * @return std::tuple<torch::Tensor, torch::Tensor>
 */
template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor> _ListSampling(torch::Tensor data,
                                                       int64_t num_picks,
                                                       bool replace) {
  int num_items = data.numel();
  torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

  if (num_items <= num_picks and !replace) {
    // todo (ping), do we need clone here?
    return std::make_tuple(data.clone(),
                           torch::arange(num_items, index_options));
  }

  torch::Tensor select;
  torch::Tensor index;

  if (replace) {
    select = torch::empty(num_picks, data.options());
    index = torch::empty(num_picks, index_options);

    uint64_t random_seed = 7777;
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        it(0), it(num_picks),
        [in = data.data_ptr<IdType>(), out_data = select.data_ptr<IdType>(),
         out_index = index.data_ptr<int64_t>(), num_items, num_picks,
         random_seed] __device__(IdType i) mutable {
          curandState rng;
          curand_init(i * random_seed, 0, 0, &rng);
          int64_t _id = curand(&rng) % num_items;
          out_data[i] = in[_id];
          out_index[i] = _id;
        });

  } else {
    index = torch::arange(num_picks, index_options);

    uint64_t random_seed = 7777;
    constexpr int BLOCK_SIZE = 256;
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_items + BLOCK_SIZE - 1) / BLOCK_SIZE);
    _ListSamplingIndexKernel<IdType>
        <<<grid, block>>>(random_seed, data.data_ptr<IdType>(),
                          index.data_ptr<int64_t>(), num_items, num_picks);

    select = data.index({index});
  }

  return std::make_tuple(select, index);
}

std::tuple<torch::Tensor, torch::Tensor> ListSampling(torch::Tensor data,
                                                      int64_t num_picks,
                                                      bool replace) {
  return _ListSampling<int64_t>(data, num_picks, replace);
}

}  // namespace impl

}  // namespace gs
