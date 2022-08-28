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
  __device__ inline NodeQueryHashmap(IdType *Kptr, bool *Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Insert(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    vptr[pos] = true;
  }

  __device__ inline bool Query(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return true;
      }
      if (kptr[pos] == kEmptyKey) {
        return false;
      }
      pos = hash(pos + delta);
      delta += 1;
    }

    return false;
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
  IdType *kptr;
  bool *vptr;
  uint32_t capacity{0};
};

template <typename IdType, int BLOCK_WARPS, int TILE_SIZE>
__global__ void _CSCRowwiseSlicinigQueryKernel(
    const IdType *const in_indptr, const IdType *const in_indices,
    IdType *const key_buffer, bool *const value_buffer, IdType *const out_deg,
    bool *const out_mask, const int num_items, const int dir_size) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  IdType last_row =
      MIN(static_cast<IdType>(blockDim.x + 1) * TILE_SIZE, num_items);

  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;

  NodeQueryHashmap<IdType> hashmap(key_buffer, value_buffer, dir_size);

  typedef cub::WarpReduce<IdType> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_WARPS];

  while (out_row < last_row) {
    IdType count = 0;
    IdType in_row_start = in_indptr[out_row];
    IdType in_row_end = in_indptr[out_row + 1];

    for (int idx = in_row_start + laneid; idx < in_row_end; idx += WARP_SIZE) {
      bool is_in = hashmap.Query(in_indices[idx]);
      count += is_in ? 1 : 0;
      out_mask[idx] = is_in;
    }

    int deg = WarpReduce(temp_storage[warp_id]).Sum(count);
    if (laneid == 0) {
      out_deg[out_row] = deg;
    }

    out_row += BLOCK_WARPS;
  }
}

// todo(ping): maybe we need to return _idxs for the selected edge;
template <typename IdType>
std::pair<torch::Tensor, torch::Tensor> _CSCRowwiseSlicing(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_ids) {
  int num_items = indptr.numel() - 1;
  int num_edge = indices.numel();
  int num_row_ids = row_ids.numel();

  // construct NodeQueryHashMap
  int dir_size = UpPower(num_row_ids);
  torch::Tensor key_buffer = torch::full(dir_size, -1, indptr.options());
  torch::Tensor value_buffer = torch::full(
      dir_size, false,
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(it(0), it(num_row_ids),
                   [key = row_ids.data_ptr<IdType>(),
                    _key_buffer = key_buffer.data_ptr<IdType>(),
                    _value_buffer = value_buffer.data_ptr<bool>(),
                    dir_size] __device__(IdType i) {
                     NodeQueryHashmap<IdType> hashmap(_key_buffer,
                                                      _value_buffer, dir_size);
                     hashmap.Insert(key[i]);
                   });

  constexpr int BLOCK_WARP = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = 16;
  const dim3 block(WARP_SIZE, BLOCK_WARP);
  const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);

  torch::Tensor out_indptr = torch::empty_like(indptr);
  torch::Tensor out_mask = torch::empty(
      num_edge,
      torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

  // query hashmap to get mask
  _CSCRowwiseSlicinigQueryKernel<IdType, BLOCK_WARP, TILE_SIZE>
      <<<grid, block>>>(indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
                        key_buffer.data_ptr<IdType>(),
                        value_buffer.data_ptr<bool>(),
                        out_indptr.data_ptr<IdType>(),
                        out_mask.data_ptr<bool>(), num_items, dir_size);

  // prefix sum to get out_indptr and out_indices_index
  cub_exclusiveSum<IdType>(out_indptr.data_ptr<IdType>(), num_items + 1);

  return {out_indptr, indices.index({out_mask})};
}

std::pair<torch::Tensor, torch::Tensor> CSCRowwiseSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_ids) {
  return _CSCRowwiseSlicing<int64_t>(indptr, indices, row_ids);
};

}  // namespace impl

}  // namespace gs