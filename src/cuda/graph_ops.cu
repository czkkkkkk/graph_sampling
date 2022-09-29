#include "graph_ops.h"

#include <curand_kernel.h>
#include <nvToolsExt.h>
#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
torch::Tensor GetSubIndptr(torch::Tensor indptr, torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  auto new_indptr = torch::zeros(size + 1, indptr.options());
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(new_indptr.data_ptr<IdType>()));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(size),
      [in = column_ids.data_ptr<IdType>(),
       in_indptr = indptr.data_ptr<IdType>(),
       out = thrust::raw_pointer_cast(item_prefix)] __device__(int i) mutable {
        IdType begin = in_indptr[in[i]];
        IdType end = in_indptr[in[i] + 1];
        out[i] = end - begin;
      });

  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), size + 1);
  return new_indptr;
}

template <typename IdType>
__global__ void _GetSubIndicesKernel(IdType* out_indices, IdType* indptr,
                                     IdType* indices, IdType* sub_indptr,
                                     IdType* column_ids, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    int64_t in_start = indptr[column_ids[row]];
    int64_t out_start = sub_indptr[row];
    int64_t n_edges = sub_indptr[row + 1] - sub_indptr[row];
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      out_indices[out_start + idx] = indices[in_start + idx];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
torch::Tensor GetSubIndices(torch::Tensor indptr, torch::Tensor indices,
                            torch::Tensor sub_indptr,
                            torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, indices.options());

  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  _GetSubIndicesKernel<int64_t><<<grid, block>>>(
      sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
      indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
      column_ids.data_ptr<int64_t>(), size);
  return sub_indices;
}

// columwise slicing
std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids) {
  auto sub_indptr = GetSubIndptr<int64_t>(indptr, column_ids);
  auto sub_indices =
      GetSubIndices<int64_t>(indptr, indices, sub_indptr, column_ids);
  return {sub_indptr, sub_indices};
}

template <int BLOCK_SIZE>
__global__ void _RandomWalkKernel(const int64_t* seed_data,
                                  const int64_t num_seeds,
                                  const uint64_t max_num_steps,
                                  int64_t* graph_indice, int64_t* graph_indptr,
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

torch::Tensor RandomWalkFusedCUDA(torch::Tensor seeds, int64_t walk_length,
                                  int64_t* indices, int64_t* indptr) {
  const int64_t* seed_data = seeds.data_ptr<int64_t>();
  const int64_t num_seeds = seeds.numel();
  const uint64_t max_num_steps = walk_length;
  int64_t outsize = num_seeds * (max_num_steps + 1);
  torch::Tensor out_traces_tensor = torch::empty(outsize, seeds.options());

  int64_t* out_traces_data = out_traces_tensor.data_ptr<int64_t>();
  constexpr int BLOCK_SIZE = 256;
  dim3 block(BLOCK_SIZE);
  dim3 grid((num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _RandomWalkKernel<BLOCK_SIZE>
      <<<grid, block>>>(seeds.data_ptr<int64_t>(), num_seeds, max_num_steps,
                        indices, all_indptr, out_traces_data);
  return out_traces_tensor.reshape({seeds.numel(), -1});
}

}  // namespace impl
}  // namespace gs