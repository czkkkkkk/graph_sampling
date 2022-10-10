#include <curand_kernel.h>
#include <nvToolsExt.h>
#include "cuda_common.h"
#include "heterograph_ops.h"
#include "utils.h"
namespace gs {
namespace impl {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseSamplingOneKeepDimCUDA(torch::Tensor indptr, torch::Tensor indices,
                                    torch::Tensor column_ids) {
  // get subptr
  int64_t num_items = column_ids.numel();
  auto sub_indptr = torch::ones(num_items + 1, indptr.options());
  thrust::device_ptr<int64_t> item_prefix(
      static_cast<int64_t*>(sub_indptr.data_ptr<int64_t>()));
  cub_exclusiveSum<int64_t>(thrust::raw_pointer_cast(item_prefix),
                            num_items + 1);
  auto select_index = torch::empty(num_items, indices.options());
  // get subindices
  auto sub_indices = torch::empty(num_items, indices.options());
  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [sub_indices_ptr = sub_indices.data_ptr<int64_t>(),
       indptr_ptr = indptr.data_ptr<int64_t>(),
       indices_ptr = indices.data_ptr<int64_t>(),
       sub_indptr_ptr = sub_indptr.data_ptr<int64_t>(),
       select_index_ptr = select_index.data_ptr<int64_t>(),
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
          select_index_ptr[out_start] = -1;
        } else {
          // Sequential Sampling
          // const int64_t edge = tid % degree;
          // Random Sampling
          const int64_t edge = curand(&rng) % degree;
          sub_indices_ptr[out_start] = indices_ptr[in_start + edge];
          select_index_ptr[out_start] = in_start + edge;
        }
      });
  return {sub_indptr, sub_indices, select_index};
}

template <int BLOCK_SIZE>
__global__ void _RandomWalkKernel(const int64_t* seed_data,
                                  const int64_t num_seeds,
                                  const int64_t* metapath_data,
                                  const uint64_t max_num_steps,
                                  int64_t** all_indices, int64_t** all_indptr,
                                  int64_t* out_traces_data) {
  int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * BLOCK_SIZE, num_seeds);
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + tid, 0, 0, &rng);
  // fisrt step
  for (int idx = tid; idx < last_idx; idx += BLOCK_SIZE) {
    int64_t curr = seed_data[idx];
    out_traces_data[0 * num_seeds + idx] = curr;
  }
  // begin random walk
  for (int step_idx = 0; step_idx < max_num_steps; step_idx++) {
    for (int idx = tid; idx < last_idx; idx += BLOCK_SIZE) {
      int64_t curr = out_traces_data[step_idx * num_seeds + idx];
      int64_t pick = -1;
      if (curr < 0) {
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      } else {
        int64_t metapath_id = metapath_data[step_idx];
        int64_t* graph_indice = all_indices[metapath_id];
        int64_t* graph_indptr = all_indptr[metapath_id];
        const int64_t in_row_start = graph_indptr[curr];
        const int64_t deg = graph_indptr[curr + 1] - graph_indptr[curr];
        if (deg > 0) {
          pick = graph_indice[in_row_start + curand(&rng) % deg];
        }
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      }
    }
  }
}

torch::Tensor MetapathRandomWalkFusedCUDA(torch::Tensor seeds,
                                          torch::Tensor metapath,
                                          int64_t** all_indices,
                                          int64_t** all_indptr) {
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
  _RandomWalkKernel<BLOCK_SIZE><<<grid, block>>>(
      seeds.data_ptr<int64_t>(), num_seeds, metapath_data, max_num_steps,
      all_indices, all_indptr, out_traces_data);
  return out_traces_tensor.reshape({seeds.numel(), -1});
}
}  // namespace impl
}  // namespace gs