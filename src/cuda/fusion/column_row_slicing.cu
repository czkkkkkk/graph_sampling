#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"
#include "column_row_slicing.h"

namespace gs {
namespace impl {
namespace fusion {

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

template <typename IdType, int BLOCK_WARPS, int TILE_SIZE>
__global__ void _RowColSlicingKernel(IdType* in_indptr, IdType* in_indices,
                                     IdType* sub_indptr, IdType* seeds,
                                     IdType* key_buffer, IdType* value_buffer,
                                     int num_items, int dir_size,
                                     IdType* out_count, IdType* out_indices,
                                     IdType* out_mask) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  IdType last_row =
      MIN(static_cast<IdType>(blockIdx.x + 1) * TILE_SIZE, num_items);

  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;

  NodeQueryHashmap<IdType> hashmap(key_buffer, value_buffer, dir_size);

  typedef cub::WarpReduce<IdType> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_WARPS];
  while (out_row < last_row) {
    IdType count = 0;
    IdType row = seeds[out_row];
    IdType in_row_start = in_indptr[row];
    IdType deg = in_indptr[row + 1] - in_row_start;
    IdType out_row_start = sub_indptr[out_row];

    for (int idx = laneid; idx < deg; idx += WARP_SIZE) {
      IdType value = hashmap.Query(in_indices[in_row_start + idx]);
      if (value != -1) {
        count += 1;
        out_mask[out_row_start + idx] = 1;
        out_indices[out_row_start + idx] = value;
      }
    }

    int out_deg = WarpReduce(temp_storage[warp_id]).Sum(count);
    if (laneid == 0) {
      out_count[out_row] = out_deg;
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename IdType>
torch::Tensor GetSubIndptr(torch::Tensor indptr, torch::Tensor seeds) {
  int64_t size = seeds.numel();
  auto new_indptr = torch::zeros(size + 1, indptr.options());

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(size),
      [in = seeds.data_ptr<IdType>(), in_indptr = indptr.data_ptr<IdType>(),
       out = new_indptr.data_ptr<IdType>()] __device__(int i) mutable {
        IdType begin = in_indptr[in[i]];
        IdType end = in_indptr[in[i] + 1];
        out[i] = end - begin;
      });

  cub_exclusiveSum<IdType>(new_indptr.data_ptr<IdType>(), size + 1);
  return new_indptr;
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _RowColSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor seeds) {
  int num_items = seeds.numel();

  // construct NodeQueryHashMap
  int dir_size = UpPower(num_items) * 2;
  torch::Tensor key_buffer = torch::full(dir_size, -1, indptr.options());
  torch::Tensor value_buffer = torch::full(dir_size, -1, indices.options());

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(it(0), it(num_items),
                   [key = seeds.data_ptr<IdType>(),
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

  // get sub_indptr
  torch::Tensor sub_indptr = GetSubIndptr<IdType>(indptr, seeds);
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int nnz = item_prefix[num_items];  // cp

  torch::Tensor out_indptr = torch::empty_like(sub_indptr);
  torch::Tensor out_indices = torch::empty(nnz, indices.options());
  torch::Tensor out_mask = torch::zeros(nnz, indices.options());

  // query hashmap to get mask
  _RowColSlicingKernel<IdType, BLOCK_WARP, TILE_SIZE><<<grid, block>>>(
      indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
      sub_indptr.data_ptr<IdType>(), seeds.data_ptr<IdType>(),
      key_buffer.data_ptr<IdType>(), value_buffer.data_ptr<IdType>(), num_items,
      dir_size, out_indptr.data_ptr<IdType>(), out_indices.data_ptr<IdType>(),
      out_mask.data_ptr<IdType>());

  // prefix sum to get out_indptr and out_indices_index
  cub_exclusiveSum<IdType>(out_indptr.data_ptr<IdType>(), num_items + 1);
  torch::Tensor select_index = torch::nonzero(out_mask).reshape({
      -1,
  });

  return {out_indptr, out_indices.index({select_index}), select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCColRowSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor seeds) {
  return _RowColSlicingCUDA<int64_t>(indptr, indices, seeds);
};

}  // namespace fusion

}  // namespace impl
}  // namespace gs