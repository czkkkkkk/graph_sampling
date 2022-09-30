#include <curand_kernel.h>
#include <nvToolsExt.h>
#include "cuda_common.h"
#include "heterograph_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSamplingOneKeepDimCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids) {
  // get subptr
  int64_t num_items = column_ids.numel();
  auto sub_indptr = torch::ones(num_items + 1, indptr.options());
  thrust::device_ptr<int64_t> item_prefix(
      static_cast<int64_t*>(sub_indptr.data_ptr<int64_t>()));
  cub_exclusiveSum<int64_t>(thrust::raw_pointer_cast(item_prefix),
                            num_items + 1);
  // get subindices
  auto sub_indices = torch::empty(num_items, indices.options());
  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [sub_indices_ptr = sub_indices.data_ptr<int64_t>(),
       indptr_ptr = indptr.data_ptr<int64_t>(),
       indices_ptr = indices.data_ptr<int64_t>(),
       sub_indptr_ptr = sub_indptr.data_ptr<int64_t>(),
       column_ids_ptr =
           column_ids.data_ptr<int64_t>()] __device__(int i) mutable {
        const uint64_t random_seed = 7777777;
        curandState rng;
        curand_init(random_seed + i, 0, 0, &rng);
        int64_t col = column_ids_ptr[i];
        int64_t in_start = indptr_ptr[col];
        int64_t out_start = sub_indptr_ptr[i];
        int64_t degree = indptr_ptr[col + 1] - indptr_ptr[col];
        if (degree == 0) {
          sub_indices_ptr[out_start] = -1;
        } else {
          // Sequential Sampling
          // const int64_t edge = tid % degree;
          // Random Sampling
          const int64_t edge = curand(&rng) % degree;
          sub_indices_ptr[out_start] = indices_ptr[in_start + edge];
        }
      });
  return {sub_indptr, sub_indices};
}

template <int BLOCK_SIZE>
__global__ void _RandomWalkKernel(const int64_t* seed_data,
                                  const int64_t num_seeds,
                                  const int64_t* metapath_data,
                                  const uint64_t max_num_steps,
                                  int64_t** all_indices, int64_t** all_indptr,
                                  int64_t* out_traces_data) {
  int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * BLOCK_SIZE, num_seeds);
  int64_t trace_length = (max_num_steps + 1);
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + idx, 0, 0, &rng);
  while (idx < last_idx) {
    int64_t curr = seed_data[idx];
    int64_t* traces_data_ptr = &out_traces_data[idx * trace_length];
    *(traces_data_ptr++) = curr;
    int64_t step_idx;
    for (step_idx = 0; step_idx < max_num_steps; ++step_idx) {
      int64_t metapath_id = metapath_data[step_idx];
      int64_t* graph_indice = all_indices[metapath_id];
      int64_t* graph_indptr = all_indptr[metapath_id];
      const int64_t in_row_start = graph_indptr[curr];
      const int64_t deg = graph_indptr[curr + 1] - graph_indptr[curr];
      if (deg == 0) {  // the degree is zero
        break;
      }
      const int64_t num = curand(&rng) % deg;
      int64_t pick = graph_indice[in_row_start + num];
      *traces_data_ptr = pick;
      ++traces_data_ptr;
      curr = pick;
    }
    for (; step_idx < max_num_steps; ++step_idx) {
      *(traces_data_ptr++) = -1;
    }
    idx += BLOCK_SIZE;
  }
}

torch::Tensor MetapathRandomWalkFusedCUDA(
    torch::Tensor seeds, torch::Tensor metapath,
    thrust::device_vector<int64_t*>& all_indices,
    thrust::device_vector<int64_t*>& all_indptr) {
  const int64_t* seed_data = seeds.data_ptr<int64_t>();
  const int64_t num_seeds = seeds.numel();
  const int64_t* metapath_data = metapath.data_ptr<int64_t>();
  const uint64_t max_num_steps = metapath.numel();
  int64_t outsize = num_seeds * (max_num_steps + 1);
  torch::Tensor out_traces_tensor = torch::empty(outsize, seeds.options());

  int64_t* out_traces_data = out_traces_tensor.data_ptr<int64_t>();
  constexpr int BLOCK_SIZE = 256;
  dim3 block(BLOCK_SIZE);
  dim3 grid((num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int64_t** all_indices_ptr = thrust::raw_pointer_cast(all_indices.data());
  int64_t** all_indptr_ptr = thrust::raw_pointer_cast(all_indptr.data());
  _RandomWalkKernel<BLOCK_SIZE><<<grid, block>>>(
      seeds.data_ptr<int64_t>(), num_seeds, metapath_data, max_num_steps,
      all_indices_ptr, all_indptr_ptr, out_traces_data);
  return out_traces_tensor.reshape({seeds.numel(), -1});
}

}  // namespace impl
}  // namespace gs