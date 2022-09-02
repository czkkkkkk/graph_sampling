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
__global__ void _SampleSubIndicesReplaceKernel(IdType* sub_indices,
                                               IdType* indptr, IdType* indices,
                                               IdType* sub_indptr, int64_t size,
                                               const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    int64_t in_start = indptr[row];
    int64_t out_start = sub_indptr[row];
    int64_t degree = indptr[row + 1] - in_start;
    int64_t fanout = sub_indptr[row + 1] - out_start;
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      const int64_t edge = curand(&rng) % degree;
      sub_indices[out_start + idx] = indices[in_start + edge];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
__global__ void _SampleSubIndicesKernel(IdType* sub_indices, IdType* indptr,
                                        IdType* indices, IdType* sub_indptr,
                                        int64_t size,
                                        const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    int64_t in_start = indptr[row];
    int64_t out_start = sub_indptr[row];
    int64_t degree = indptr[row + 1] - in_start;
    int64_t fanout = sub_indptr[row + 1] - out_start;
    if (degree <= fanout) {
      for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        sub_indices[out_start + idx] = indices[in_start + idx];
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
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
torch::Tensor SampleSubIndices(torch::Tensor indptr, torch::Tensor indices,
                               torch::Tensor sub_indptr, bool replace) {
  int64_t size = sub_indptr.numel() - 1;
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, indices.options());

  const uint64_t random_seed = 7777;
  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  if (replace) {
    _SampleSubIndicesReplaceKernel<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(), size,
        random_seed);
  } else {
    _SampleSubIndicesKernel<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(), size,
        random_seed);
  }
  return sub_indices;
}

// columnwise sampling
std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSamplingCUDA(
    torch::Tensor indptr, torch::Tensor indices, int64_t fanout, bool replace) {
  auto sub_indptr = GetSampledSubIndptr<int64_t>(indptr, fanout, replace);
  auto sub_indices =
      SampleSubIndices<int64_t>(indptr, indices, sub_indptr, replace);
  return {sub_indptr, sub_indices};
}

template <typename IdType>
__global__ void _SampleSubIndicesReplaceKernelFused(
    IdType* sub_indices, IdType* indptr, IdType* indices, IdType* sub_indptr,
    IdType* column_ids, int64_t size, const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    int64_t col = column_ids[row];
    int64_t in_start = indptr[col];
    int64_t out_start = sub_indptr[row];
    int64_t degree = indptr[col + 1] - indptr[col];
    int64_t fanout = sub_indptr[row + 1] - sub_indptr[row];
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      const int64_t edge = curand(&rng) % degree;
      sub_indices[out_start + idx] = indices[in_start + edge];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
__global__ void _SampleSubIndicesKernelFused(IdType* sub_indices,
                                             IdType* indptr, IdType* indices,
                                             IdType* sub_indptr,
                                             IdType* column_ids, int64_t size,
                                             const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    int64_t col = column_ids[row];
    int64_t in_start = indptr[col];
    int64_t out_start = sub_indptr[row];
    int64_t degree = indptr[col + 1] - indptr[col];
    int64_t fanout = sub_indptr[row + 1] - sub_indptr[row];
    if (degree <= fanout) {
      for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        sub_indices[out_start + idx] = indices[in_start + idx];
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
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
torch::Tensor SampleSubIndicesFused(torch::Tensor indptr, torch::Tensor indices,
                                    torch::Tensor sub_indptr,
                                    torch::Tensor column_ids, bool replace) {
  // nvtxRangePush(__FUNCTION__);
  // nvtxMark("==SampleSubIndicesFused==");
  int64_t size = sub_indptr.numel() - 1;
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, indices.options());

  const uint64_t random_seed = 7777;
  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  if (replace) {
    _SampleSubIndicesReplaceKernelFused<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
        column_ids.data_ptr<int64_t>(), size, random_seed);
  } else {
    _SampleSubIndicesKernelFused<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
        column_ids.data_ptr<int64_t>(), size, random_seed);
  }
  // nvtxRangePop();
  return sub_indices;
}

// Fused columnwise slicing and sampling
std::pair<torch::Tensor, torch::Tensor>
CSCColumnwiseFusedSlicingAndSamplingCUDA(torch::Tensor indptr,
                                         torch::Tensor indices,
                                         torch::Tensor column_ids,
                                         int64_t fanout, bool replace) {
  auto sub_indptr =
      GetSampledSubIndptrFused<int64_t>(indptr, column_ids, fanout, replace);
  auto sub_indices = SampleSubIndicesFused<int64_t>(indptr, indices, sub_indptr,
                                                    column_ids, replace);
  return {sub_indptr, sub_indices};
}

}  // namespace impl
}  // namespace gs