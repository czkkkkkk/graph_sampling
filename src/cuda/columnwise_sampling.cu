#include "graph_ops.h"

#include <curand_kernel.h>
#include <nvToolsExt.h>
#include "atomic.h"
#include "cuda_common.h"
#include "sampling_utils.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
__global__ void _SampleSubIndicesWithEIDReplaceKernel(
    IdType* sub_indices, IdType* sub_e_ids, IdType* indptr, IdType* indices,
    IdType* e_ids, IdType* sub_indptr, int64_t size,
    const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    IdType in_start = indptr[row];
    IdType out_start = sub_indptr[row];
    IdType degree = indptr[row + 1] - in_start;
    IdType fanout = sub_indptr[row + 1] - out_start;
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      const IdType edge = curand(&rng) % degree;
      sub_indices[out_start + idx] = indices[in_start + edge];
      sub_e_ids[out_start + idx] = e_ids[in_start + edge];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
__global__ void _SampleSubIndicesWithEIDKernel(IdType* sub_indices,
                                               IdType* sub_e_ids,
                                               IdType* indptr, IdType* indices,
                                               IdType* e_ids,
                                               IdType* sub_indptr, int64_t size,
                                               const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    IdType in_start = indptr[row];
    IdType out_start = sub_indptr[row];
    IdType degree = indptr[row + 1] - in_start;
    IdType fanout = sub_indptr[row + 1] - out_start;
    if (degree <= fanout) {
      for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        sub_indices[out_start + idx] = indices[in_start + idx];
        sub_e_ids[out_start + idx] = e_ids[in_start + idx];
      }
    } else {
      // reservoir algorithm
      for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
        sub_indices[out_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = fanout + threadIdx.x; idx < degree; idx += blockDim.x) {
        const int num = curand(&rng) % (idx + 1);
        if (num < fanout) {
          AtomicMax(sub_indices + out_start + num, IdType(idx));
        }
      }
      __syncthreads();

      for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
        const IdType perm_idx = in_start + sub_indices[out_start + idx];
        sub_indices[out_start + idx] = indices[perm_idx];
        sub_e_ids[out_start + idx] = e_ids[perm_idx];
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
std::pair<torch::Tensor, torch::Tensor> SampleSubIndices(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor e_ids,
    torch::Tensor sub_indptr, bool replace) {
  int64_t size = sub_indptr.numel() - 1;
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, indices.options());
  auto sub_e_ids = torch::zeros(n_edges, e_ids.options());

  const uint64_t random_seed = 7777;
  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  if (replace) {
    _SampleSubIndicesWithEIDReplaceKernel<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), sub_e_ids.data_ptr<int64_t>(),
        indptr.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
        e_ids.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(), size,
        random_seed);
  } else {
    _SampleSubIndicesWithEIDKernel<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), sub_e_ids.data_ptr<int64_t>(),
        indptr.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
        e_ids.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(), size,
        random_seed);
  }
  return {sub_indices, sub_e_ids};
}

// columnwise sampling
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseSamplingCUDA(torch::Tensor indptr, torch::Tensor indices,
                          torch::Tensor e_ids, int64_t fanout, bool replace) {
  torch::Tensor sub_indptr, sub_indices, sub_e_ids;
  sub_indptr = GetSampledSubIndptr<int64_t>(indptr, fanout, replace);
  std::tie(sub_indices, sub_e_ids) =
      SampleSubIndices<int64_t>(indptr, indices, e_ids, sub_indptr, replace);
  return {sub_indptr, sub_indices, sub_e_ids};
}

template <typename IdType>
__global__ void _SampleSubIndicesWithEIDReplaceKernelFused(
    IdType* sub_indices, IdType* sub_e_ids, IdType* indptr, IdType* indices,
    IdType* e_ids, IdType* sub_indptr, IdType* column_ids, int64_t size,
    const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    IdType col = column_ids[row];
    IdType in_start = indptr[col];
    IdType out_start = sub_indptr[row];
    IdType degree = indptr[col + 1] - indptr[col];
    IdType fanout = sub_indptr[row + 1] - sub_indptr[row];
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      const IdType edge = curand(&rng) % degree;
      sub_indices[out_start + idx] = indices[in_start + edge];
      sub_e_ids[out_start + idx] = e_ids[in_start + edge];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
__global__ void _SampleSubIndicesWithEIDKernelFused(IdType* sub_indices,
                                             IdType* sub_e_ids, IdType* indptr,
                                             IdType* indices, IdType* e_ids,
                                             IdType* sub_indptr,
                                             IdType* column_ids, int64_t size,
                                             const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    IdType col = column_ids[row];
    IdType in_start = indptr[col];
    IdType out_start = sub_indptr[row];
    IdType degree = indptr[col + 1] - indptr[col];
    IdType fanout = sub_indptr[row + 1] - sub_indptr[row];
    if (degree <= fanout) {
      for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        sub_indices[out_start + idx] = indices[in_start + idx];
        sub_e_ids[out_start + idx] = e_ids[in_start + idx];
      }
    } else {
      // reservoir algorithm
      for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
        sub_indices[out_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = fanout + threadIdx.x; idx < degree; idx += blockDim.x) {
        const int num = curand(&rng) % (idx + 1);
        if (num < fanout) {
          AtomicMax(sub_indices + out_start + num, IdType(idx));
        }
      }
      __syncthreads();

      for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
        const IdType perm_idx = in_start + sub_indices[out_start + idx];
        sub_indices[out_start + idx] = indices[perm_idx];
        sub_e_ids[out_start + idx] = e_ids[perm_idx];
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
std::pair<torch::Tensor, torch::Tensor> SampleSubIndicesFused(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor e_ids,
    torch::Tensor sub_indptr, torch::Tensor column_ids, bool replace) {
  int64_t size = sub_indptr.numel() - 1;
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, indices.options());
  auto sub_e_ids = torch::zeros(n_edges, e_ids.options());

  const uint64_t random_seed = 7777;
  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  if (replace) {
    _SampleSubIndicesWithEIDReplaceKernelFused<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), sub_e_ids.data_ptr<int64_t>(),
        indptr.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
        e_ids.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
        column_ids.data_ptr<int64_t>(), size, random_seed);
  } else {
    _SampleSubIndicesWithEIDKernelFused<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), sub_e_ids.data_ptr<int64_t>(),
        indptr.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
        e_ids.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
        column_ids.data_ptr<int64_t>(), size, random_seed);
  }
  return {sub_indices, sub_e_ids};
}

// Fused columnwise slicing and sampling
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseFusedSlicingAndSamplingCUDA(torch::Tensor indptr,
                                         torch::Tensor indices,
                                         torch::Tensor e_ids,
                                         torch::Tensor column_ids,
                                         int64_t fanout, bool replace) {
  torch::Tensor sub_indptr, sub_indices, sub_e_ids;
  sub_indptr =
      GetSampledSubIndptrFused<int64_t>(indptr, column_ids, fanout, replace);
  std::tie(sub_indices, sub_e_ids) = SampleSubIndicesFused<int64_t>(
      indptr, indices, e_ids, sub_indptr, column_ids, replace);
  return {sub_indptr, sub_indices, sub_e_ids};
}

}  // namespace impl
}  // namespace gs