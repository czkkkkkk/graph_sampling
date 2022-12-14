#include "atomic.h"
#include "cuda_common.h"
#include "graph_ops.h"
#include "utils.h"

namespace gs {
namespace impl {
inline __host__ __device__ int UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

__device__ inline uint32_t Hash32Shift(uint32_t key) {
  key = ~key + (key << 15);  // # key = (key << 15) - key - 1;
  key = key ^ (key >> 12);
  key = key + (key << 2);
  key = key ^ (key >> 4);
  key = key * 2057;  // key = (key + (key << 3)) + (key << 11);
  key = key ^ (key >> 16);
  return key;
}

__device__ inline uint64_t Hash64Shift(uint64_t key) {
  key = (~key) + (key << 21);  // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8);  // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4);  // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

/**
 * @brief Used to judge whether a node is in a node set
 *
 * @tparam IdType
 */
template <typename IdType>
struct NodeQueryHashmap {
  __device__ inline NodeQueryHashmap(IdType* Kptr, IdType* Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Insert(IdType key, IdType value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    vptr[pos] = value;
  }

  __device__ inline IdType Query(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return vptr[pos];
      }
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }

    return -1;
  }

  __device__ inline uint32_t hash(int32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(int64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  IdType kEmptyKey{-1};
  IdType* kptr;
  IdType* vptr;
  uint32_t capacity{0};
};

////////////////////////////// indptr slicing ///////////////////////////
template <typename IdType>
__global__ void _GetSubIndicesKernel(IdType* out_indices, IdType* select_index,
                                     IdType* out_row, IdType* indptr,
                                     IdType* indices, IdType* sub_indptr,
                                     IdType* column_ids, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  while (row < size) {
    IdType in_start = indptr[column_ids[row]];
    IdType out_start = sub_indptr[row];
    IdType n_edges = sub_indptr[row + 1] - out_start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      out_indices[out_start + idx] = indices[in_start + idx];
      select_index[out_start + idx] = in_start + idx;
      out_row[out_start + idx] = column_ids[row];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _OnIndptrSlicing(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids) {
  int64_t num_items = column_ids.numel();

  // compute indptr
  torch::Tensor sub_indptr = torch::empty(num_items + 1, indptr.options());
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [in = column_ids.data_ptr<IdType>(),
       in_indptr = indptr.data_ptr<IdType>(),
       out = sub_indptr.data_ptr<IdType>()] __device__(int i) mutable {
        IdType begin = in_indptr[in[i]];
        IdType end = in_indptr[in[i] + 1];
        out[i] = end - begin;
      });
  cub_exclusiveSum<IdType>(sub_indptr.data_ptr<IdType>(), num_items + 1);

  // compute indices
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int nnz = item_prefix[num_items];  // cpu
  torch::Tensor out_coo_row = torch::empty(nnz, indices.options());
  torch::Tensor out_coo_col = torch::empty(nnz, indices.options());
  torch::Tensor select_index = torch::empty(nnz, indices.options());

  dim3 block(32, 16);
  dim3 grid((num_items + block.x - 1) / block.x);
  _GetSubIndicesKernel<IdType><<<grid, block>>>(
      out_coo_row.data_ptr<IdType>(), select_index.data_ptr<IdType>(),
      out_coo_col.data_ptr<IdType>(), indptr.data_ptr<IdType>(),
      indices.data_ptr<IdType>(), sub_indptr.data_ptr<IdType>(),
      column_ids.data_ptr<IdType>(), num_items);
  return {out_coo_row, out_coo_col, select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FullCSCColSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids) {
  torch::Tensor coo_row, coo_col, selected_index;
  std::tie(coo_row, coo_col, selected_index) =
      _OnIndptrSlicing<int64_t>(indptr, indices, column_ids);
  return {coo_row, coo_col, selected_index};
}

////////////////////////////// indices slicing //////////////////////////
template <typename IdType, int BLOCK_WARPS, int TILE_SIZE>
__global__ void _OnIndicesSlicinigQueryKernel(
    const IdType* const in_indptr, const IdType* const in_indices,
    IdType* const key_buffer, IdType* const value_buffer, IdType* out_coo_col,
    IdType* const out_mask, const int num_items, const int dir_size) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  IdType last_row =
      MIN(static_cast<IdType>(blockIdx.x + 1) * TILE_SIZE, num_items);

  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;

  NodeQueryHashmap<IdType> hashmap(key_buffer, value_buffer, dir_size);

  while (out_row < last_row) {
    IdType in_row_start = in_indptr[out_row];
    IdType in_row_end = in_indptr[out_row + 1];

    for (int idx = in_row_start + laneid; idx < in_row_end; idx += WARP_SIZE) {
      IdType value = hashmap.Query(in_indices[idx]);
      if (value != -1) {
        out_mask[idx] = 1;
        out_coo_col[idx] = out_row;
      }
    }
    out_row += BLOCK_WARPS;
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _OnIndicesSlicing(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_ids) {
  int num_items = indptr.numel() - 1;
  int num_row_ids = row_ids.numel();

  // construct NodeQueryHashMap
  int dir_size = UpPower(num_row_ids) * 2;
  torch::Tensor key_buffer = torch::full(dir_size, -1, indptr.options());
  torch::Tensor value_buffer = torch::full(dir_size, -1, indices.options());
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_row_ids),
                   [key = row_ids.data_ptr<IdType>(),
                    _key_buffer = key_buffer.data_ptr<IdType>(),
                    _value_buffer = value_buffer.data_ptr<IdType>(),
                    dir_size] __device__(IdType i) {
                     NodeQueryHashmap<IdType> hashmap(_key_buffer,
                                                      _value_buffer, dir_size);
                     hashmap.Insert(key[i], i);
                   });

  constexpr int BLOCK_WARP = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = 16;
  const dim3 block(WARP_SIZE, BLOCK_WARP);
  const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);

  torch::Tensor out_coo_col = torch::empty_like(indices);
  torch::Tensor out_mask = torch::zeros_like(indices);
  // query hashmap to get mask
  _OnIndicesSlicinigQueryKernel<IdType, BLOCK_WARP, TILE_SIZE><<<grid, block>>>(
      indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
      key_buffer.data_ptr<IdType>(), value_buffer.data_ptr<IdType>(),
      out_coo_col.data_ptr<IdType>(), out_mask.data_ptr<IdType>(), num_items,
      dir_size);

  torch::Tensor select_index = torch::nonzero(out_mask).reshape({
      -1,
  });

  return {indices.index({select_index}), out_coo_col.index({select_index}),
          select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FullCSCRowSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_ids) {
  torch::Tensor coo_row, coo_col, select_index;

  std::tie(coo_row, coo_col, select_index) =
      _OnIndicesSlicing<int64_t>(indptr, indices, row_ids);
  return {coo_row, coo_col, select_index};
};

////////////////////////////// COORowSlicingCUDA //////////////////////////
// reuse hashmap in CSCRowSlicingCUDA
template <typename IdType>
__global__ void _COORowSlicingKernel(const IdType* const in_coo_row,
                                     IdType* const key_buffer,
                                     IdType* const value_buffer,
                                     IdType* const out_mask,
                                     const int num_items, const int dir_size) {
  IdType tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdType stride = gridDim.x * blockDim.x;

  NodeQueryHashmap<IdType> hashmap(key_buffer, value_buffer, dir_size);

  while (tid < num_items) {
    IdType value = hashmap.Query(in_coo_row[tid]);
    if (value != -1) {
      out_mask[tid] = 1;
    }
    tid += stride;
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _COORowSlicing(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids) {
  int num_items = coo_row.numel();
  int num_row_ids = row_ids.numel();

  // construct NodeQueryHashMap
  int dir_size = UpPower(num_row_ids) * 2;
  torch::Tensor key_buffer = torch::full(dir_size, -1, row_ids.options());
  torch::Tensor value_buffer = torch::full(dir_size, -1, row_ids.options());

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_row_ids),
                   [key = row_ids.data_ptr<IdType>(),
                    _key_buffer = key_buffer.data_ptr<IdType>(),
                    _value_buffer = value_buffer.data_ptr<IdType>(),
                    dir_size] __device__(IdType i) {
                     NodeQueryHashmap<IdType> hashmap(_key_buffer,
                                                      _value_buffer, dir_size);
                     hashmap.Insert(key[i], i);
                   });

  torch::Tensor out_mask = torch::zeros_like(coo_row);

  constexpr int TILE_SIZE = 16;
  const dim3 block(256);
  const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);

  _COORowSlicingKernel<IdType><<<grid, block>>>(
      coo_row.data_ptr<IdType>(), key_buffer.data_ptr<IdType>(),
      value_buffer.data_ptr<IdType>(), out_mask.data_ptr<IdType>(), num_items,
      dir_size);

  torch::Tensor select_index = torch::nonzero(out_mask).reshape({
      -1,
  });
  return {coo_row.index({select_index}), coo_col.index({select_index}),
          select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FullCOORowSlicingCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids) {
  return _COORowSlicing<int64_t>(coo_row, coo_col, row_ids);
};

}  // namespace impl
}  // namespace gs