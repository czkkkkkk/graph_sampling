#include <curand_kernel.h>
#include <tuple>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType, typename FloatType>
std::tuple<torch::Tensor, torch::Tensor> _ListSamplingProbs(torch::Tensor data,
                                                            torch::Tensor probs,
                                                            int64_t num_picks,
                                                            bool replace) {
  int num_items = data.numel();
  assert(data.numel() == probs.numel());
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
    // using cdf sampling
    torch::Tensor prefix_probs = probs.clone();
    select = torch::empty(num_picks, data.options());
    index = torch::empty(num_picks, index_options);

    // prefix_sum
    cub_inclusiveSum<FloatType>(probs.data_ptr<FloatType>(), num_items);

    uint64_t random_seed = 7777;
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        it(0), it(num_picks),
        [_in = data.data_ptr<IdType>(), _index = index.data_ptr<int64_t>(),
         _prefix_probs = probs.data_ptr<FloatType>(),
         _out = select.data_ptr<IdType>(), num_items,
         random_seed] __device__(IdType i) mutable {
          curandState rng;
          curand_init(i * random_seed, 0, 0, &rng);
          FloatType sum = _prefix_probs[num_items - 1];
          FloatType rand = static_cast<FloatType>(curand_uniform(&rng) * sum);
          int64_t item = cub::UpperBound<FloatType*, int64_t, FloatType>(
              _prefix_probs, num_items, rand);
          item = MIN(item, num_items - 1);
          // output
          _out[i] = _in[item];
          _index[i] = item;
        });
  } else {
    // using A-Res sampling
    torch::Tensor ares_tensor = torch::empty_like(probs);
    torch::Tensor ares_index = torch::empty(num_items, index_options);
    uint64_t random_seed = 7777;
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(it(0), it(num_items),
                     [_probs = probs.data_ptr<FloatType>(),
                      _ares = ares_tensor.data_ptr<FloatType>(),
                      _ares_ids = ares_index.data_ptr<int64_t>(),
                      random_seed] __device__(IdType i) mutable {
                       curandState rng;
                       curand_init(i * random_seed, 0, 0, &rng);
                       FloatType item_prob = _probs[i];
                       FloatType ares_prob =
                           __powf(curand_uniform(&rng), 1.0f / item_prob);
                       _ares[i] = ares_prob;
                       _ares_ids[i] = i;
                     });

    torch::Tensor sort_ares = torch::empty_like(ares_tensor);
    torch::Tensor sort_index = torch::empty_like(ares_index);

    cub::DoubleBuffer<FloatType> d_keys(ares_tensor.data_ptr<FloatType>(),
                                        sort_ares.data_ptr<FloatType>());
    cub::DoubleBuffer<int64_t> d_values(ares_index.data_ptr<int64_t>(),
                                        sort_index.data_ptr<int64_t>());

    cub_sortPairsDescending<FloatType, int64_t>(d_keys, d_values, num_items);

    index = sort_index.slice(0, 0, num_picks, 1);
    select = data.index({index});
  }

  return std::make_tuple(select, index);
}

std::tuple<torch::Tensor, torch::Tensor> ListSamplingProbsCUDA(
    torch::Tensor data, torch::Tensor probs, int64_t num_picks, bool replace) {
  CHECK(data.dtype() == torch::kInt64);
  CHECK(probs.dtype() == torch::kFloat);
  return _ListSamplingProbs<int64_t, float>(data, probs, num_picks, replace);
}

}  // namespace impl
}  // namespace gs