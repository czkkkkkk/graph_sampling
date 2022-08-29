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
    int64_t tid = threadIdx.x;
    while (tid < n_edges) {
      out_indices[out_start + tid] = indices[in_start + tid];
      tid += blockDim.x;
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

  dim3 block(32, 8);
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

template <typename IdType>
torch::Tensor GetSampledSubIndptr(torch::Tensor indptr, int64_t fanout,
                                  bool replace) {
  int64_t size = indptr.numel();
  auto new_indptr = torch::zeros(size, indptr.options());
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(new_indptr.data_ptr<IdType>()));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(size),
      [in_indptr = indptr.data_ptr<IdType>(),
       out = thrust::raw_pointer_cast(item_prefix), if_replace = replace,
       num_fanout = fanout] __device__(int i) mutable {
        IdType begin = in_indptr[i];
        IdType end = in_indptr[i + 1];
        if (if_replace) {
          out[i] = (end - begin) == 0 ? 0 : num_fanout;
        } else {
          out[i] = min(end - begin, num_fanout);
        }
      });

  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), size + 1);
  return new_indptr;
}

template <typename IdType>
__global__ void _SampleSubIndicesKernel(IdType* sub_indices, IdType* indptr,
                                        IdType* indices, IdType* sub_indptr,
                                        int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  const uint64_t random_seed = 7777777;
  curandState rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
  while (row < size) {
    int64_t in_start = indptr[row];
    int64_t out_start = sub_indptr[row];
    int64_t degree = indptr[row + 1] - in_start;
    int64_t fanout = sub_indptr[row + 1] - out_start;
    int64_t tid = threadIdx.x;
    while (tid < fanout) {
      // Sequential Sampling
      // const int64_t edge = tid % degree;
      // Random Sampling
      const int64_t edge = curand(&rng) % degree;
      sub_indices[out_start + tid] = indices[in_start + edge];
      tid += blockDim.x;
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
torch::Tensor SampleSubIndices(torch::Tensor indptr, torch::Tensor indices,
                               torch::Tensor sub_indptr) {
  int64_t size = sub_indptr.numel() - 1;
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, indices.options());

  dim3 block(32, 8);
  dim3 grid((size + block.x - 1) / block.x);
  _SampleSubIndicesKernel<int64_t><<<grid, block>>>(
      sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
      indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(), size);
  return sub_indices;
}

// columnwise sampling
std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSamplingCUDA(
    torch::Tensor indptr, torch::Tensor indices, int64_t fanout, bool replace) {
  auto sub_indptr = GetSampledSubIndptr<int64_t>(indptr, fanout, replace);
  auto sub_indices = SampleSubIndices<int64_t>(indptr, indices, sub_indptr);
  return {sub_indptr, sub_indices};
}

template <typename IdType>
torch::Tensor GetSampledSubIndptrFused(torch::Tensor indptr,
                                       torch::Tensor column_ids, int64_t fanout,
                                       bool replace) {
  int64_t size = column_ids.numel();
  auto sub_indptr = torch::empty(size + 1, indptr.options());
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(size),
      [in = column_ids.data_ptr<IdType>(),
       in_indptr = indptr.data_ptr<IdType>(),
       out = thrust::raw_pointer_cast(item_prefix), if_replace = replace,
       num_fanout = fanout] __device__(int i) mutable {
        IdType begin = in_indptr[in[i]];
        IdType end = in_indptr[in[i] + 1];
        if (if_replace) {
          out[i] = (end - begin) == 0 ? 0 : num_fanout;
        } else {
          out[i] = min(end - begin, num_fanout);
        }
      });

  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), size + 1);
  return sub_indptr;
}

template <typename IdType>
__global__ void _SampleSubIndicesKernelFusedWithReplace(
    IdType* sub_indices, IdType* indptr, IdType* indices, IdType* sub_indptr,
    IdType* column_ids, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  const uint64_t random_seed = 7777777;
  curandState rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
  while (row < size) {
    int64_t col = column_ids[row];
    int64_t in_start = indptr[col];
    int64_t out_start = sub_indptr[row];
    int64_t degree = indptr[col + 1] - indptr[col];
    int64_t fanout = sub_indptr[row + 1] - sub_indptr[row];
    int64_t tid = threadIdx.x;
    while (tid < fanout) {
      // Sequential Sampling
      // const int64_t edge = tid % degree;
      // Random Sampling
      const int64_t edge = curand(&rng) % degree;
      sub_indices[out_start + tid] = indices[in_start + edge];
      tid += blockDim.x;
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

  if (replace) {
    dim3 block(32, 8);
    dim3 grid((size + block.x - 1) / block.x);
    _SampleSubIndicesKernelFusedWithReplace<int64_t><<<grid, block>>>(
        sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
        indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
        column_ids.data_ptr<int64_t>(), size);
  } else {
    std::cerr << "Not implemented warning";
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
  auto sub_indptr = GetSampledSubIndptrFused<int64_t>(indptr, column_ids, fanout, replace);
  auto sub_indices = SampleSubIndicesFused<int64_t>(indptr, indices, sub_indptr,
                                                    column_ids, replace);
  return {sub_indptr, sub_indices};
}

template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void _RandomWalkKernel(
    const int64_t* seed_data, const int64_t num_seeds,
    const int64_t* metapath_data, const uint64_t max_num_steps,
    int64_t* homo_indptr,
    int64_t* homo_indice, 
    int64_t* homo_indptr_offsets,
    int64_t* homo_indices_offsets,
    int64_t *out_traces_data) {
  int64_t idx = blockIdx.x * TILE_SIZE + threadIdx.x;
  int64_t last_idx = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_seeds);
  int64_t trace_length = (max_num_steps + 1);
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + idx, 0, 0, &rng);

  while (idx < last_idx) {  
    int64_t curr = seed_data[idx];
    int64_t *traces_data_ptr = &out_traces_data[idx * trace_length];
    *(traces_data_ptr++) = curr;
    int64_t step_idx;
    for (step_idx = 0; step_idx < max_num_steps; ++step_idx) {
      int64_t metapath_id = metapath_data[step_idx];
      int64_t indptr_offset = homo_indptr_offsets[metapath_id];
      int64_t indices_offset = homo_indices_offsets[metapath_id]; 
      const int64_t in_row_start = homo_indptr[indptr_offset+curr];
      const int64_t deg = homo_indptr[indptr_offset+ curr + 1] - homo_indptr[indptr_offset+curr];
      if (deg == 0) {  // the degree is zero
        break;
      }
      const int64_t num = curand(&rng) % deg;
      int64_t pick = homo_indice[indices_offset + in_row_start + num];
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
//not implemented
torch::Tensor MetapathRandomWalkFusedCUDA(torch::Tensor seeds,
  torch::Tensor metapath,
  torch::Tensor homo_indptr_tensor,
  torch::Tensor homo_indice_tensor, 
  torch::Tensor homo_indptr_offset_tensor,
  torch::Tensor homo_indices_offset_tensor){
    const int64_t *seed_data = seeds.data_ptr<int64_t>(); 
    const int64_t num_seeds = seeds.numel();
    const int64_t* metapath_data = metapath.data_ptr<int64_t>(); 
    const uint64_t max_num_steps = metapath.numel();
    int64_t* homo_indptr = homo_indptr_tensor.data_ptr<int64_t>();
    int64_t* homo_indice = homo_indice_tensor.data_ptr<int64_t>();
    int64_t* homo_indptr_offsets = homo_indptr_offset_tensor.data_ptr<int64_t>();
    int64_t* homo_indices_offsets = homo_indices_offset_tensor.data_ptr<int64_t>();
    int64_t outsize = num_seeds*(max_num_steps+1);
    torch::Tensor out_traces_tensor = torch::full({outsize,},-1, seeds.options()).to(torch::kCUDA);
    int64_t *out_traces_data = out_traces_tensor.data_ptr<int64_t>();
    constexpr int BLOCK_SIZE = 256;
    constexpr int TILE_SIZE = BLOCK_SIZE * 4;
    dim3 block(256);
    dim3 grid((num_seeds + TILE_SIZE - 1) / TILE_SIZE);
    std::cout<<__FILE__<<__LINE__<<std::endl;
    _RandomWalkKernel<BLOCK_SIZE ,TILE_SIZE><<<grid,block>>>(seed_data,num_seeds,
      metapath_data,max_num_steps,
      homo_indptr,
      homo_indice,
      homo_indptr_offsets,
      homo_indices_offsets,
      out_traces_data);
      std::cout<<__FILE__<<__LINE__<<std::endl;
  return out_traces_tensor;
    }
}  // namespace impl
}  // namespace gs