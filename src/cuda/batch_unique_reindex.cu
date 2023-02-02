#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
struct RelabelHashmap {
  __device__ inline RelabelHashmap(IdType* Kptr, IdType* Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Update(IdType key, IdType value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    AtomicMin(vptr + pos, value);
  }

  __device__ inline IdType SearchForPos(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return pos;
      }
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline IdType SearchForValue(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return vptr[pos];
      };
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline uint32_t hash(int32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(int64_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint64_t key) { return key & (capacity - 1); }

  IdType kEmptyKey{-1};
  IdType* kptr;
  IdType* vptr;
  uint32_t capacity{0};
};

inline int UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

// each block work for one batch
template <typename IdType, int BLOCK_SIZE>
__global__ void _InsertHashmaps(IdType* key_tensor, IdType* value_tensor,
                                IdType* hashmap_ptr, IdType* prefix_tensor,
                                IdType* prefix_tensor_ptr,
                                IdType* unique_tensor_split_size,
                                IdType** batch_tensors, IdType** segments_ptr,
                                int64_t num_batchs, int64_t num_segments) {
  int tid = threadIdx.x;
  int thread_stride = blockDim.x;
  assert(thread_stride == BLOCK_SIZE);

  int bid = blockIdx.x;
  int block_stride = gridDim.x;

  typedef cub::BlockScan<IdType, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  for (int i = bid; i < num_batchs; i += block_stride) {
    int64_t hashmap_begin = hashmap_ptr[i];
    int64_t dir_size = hashmap_ptr[i + 1] - hashmap_begin;
    RelabelHashmap<IdType> table(key_tensor + hashmap_begin,
                                 value_tensor + hashmap_begin, dir_size);

    int64_t scan_count = 0;
    for (int j = 0; j < num_segments; j++) {
      int64_t data_begin = segments_ptr[j][i];
      int64_t data_end = segments_ptr[j][i + 1];
      for (int k = tid; k < data_end - data_begin; k += thread_stride) {
        table.Update(batch_tensors[j][k + data_begin], k + scan_count);
      }
      scan_count += data_end - data_begin;
    }
    __syncthreads();

    scan_count = 0;
    int64_t out_begin = prefix_tensor_ptr[i];
    int64_t out_end = prefix_tensor_ptr[i + 1];
    for (int j = 0; j < num_segments; j++) {
      int64_t data_begin = segments_ptr[j][i];
      int64_t data_end = segments_ptr[j][i + 1];
      for (int k = tid; k < data_end - data_begin; k += thread_stride) {
        prefix_tensor[k + scan_count + out_begin] =
            table.SearchForValue(batch_tensors[j][k + data_begin]) ==
                    k + scan_count
                ? 1
                : 0;
      }
      scan_count += data_end - data_begin;
    }
    __syncthreads();

    // prefix sum
    int prefix_sum_len = out_end - out_begin;
    int upper_bound = prefix_sum_len / BLOCK_SIZE * BLOCK_SIZE;
    int64_t thread_data = 0;
    int64_t block_aggregate = 0;
    for (int k = tid; k < upper_bound; k += thread_stride) {
      thread_data = tid == 0 ? prefix_tensor[k + out_begin] + block_aggregate
                             : prefix_tensor[k + out_begin];
      thread_data = prefix_tensor[k + out_begin];
      BlockScan(temp_storage)
          .ExclusiveSum(thread_data, thread_data, block_aggregate);
      __syncthreads();
      prefix_tensor[k + out_begin] = thread_data;
    }

    thread_data = 0;
    if (tid < prefix_sum_len - upper_bound) {
      thread_data = tid == 0 ? prefix_tensor[tid + upper_bound + out_begin] +
                                   block_aggregate
                             : prefix_tensor[tid + upper_bound + out_begin];
    }
    BlockScan(temp_storage)
        .ExclusiveSum(thread_data, thread_data, block_aggregate);
    __syncthreads();

    if (tid < prefix_sum_len - upper_bound) {
      prefix_tensor[tid + upper_bound + out_begin] = thread_data;
    }

    if (tid == 0) {
      unique_tensor_split_size[i] = block_aggregate;
    }
  }
}

// each block work for one batch
template <typename IdType, int BLOCK_SIZE>
__global__ void _UpdateHashmaps(
    IdType* key_tensor, IdType* index_tensor, IdType* value_tensor,
    IdType* hashmap_ptr, IdType* prefix_tensor, IdType* prefix_tensor_ptr,
    IdType* unique_tensor, IdType* unique_tensor_ptr, IdType** batch_tensors,
    IdType** segments_ptr, int64_t num_batchs, int64_t num_segments) {
  int tid = threadIdx.x;
  int thread_stride = blockDim.x;
  assert(thread_stride == BLOCK_SIZE);

  int bid = blockIdx.x;
  int block_stride = gridDim.x;

  for (int i = bid; i < num_batchs; i += block_stride) {
    int64_t hashmap_begin = hashmap_ptr[i];
    int64_t dir_size = hashmap_ptr[i + 1] - hashmap_begin;
    RelabelHashmap<IdType> table(key_tensor + hashmap_begin,
                                 index_tensor + hashmap_begin, dir_size);

    int64_t scan_count = 0;
    int64_t unique_tensor_begin = unique_tensor_ptr[i];
    int64_t prefix_tensor_begin = prefix_tensor_ptr[i];
    for (int j = 0; j < num_segments; j++) {
      int64_t data_begin = segments_ptr[j][i];
      int64_t data_end = segments_ptr[j][i + 1];
      for (int k = tid; k < data_end - data_begin; k += thread_stride) {
        IdType pos = table.SearchForPos(batch_tensors[j][k + data_begin]);
        if (index_tensor[pos + hashmap_begin] == k + scan_count) {
          value_tensor[pos + hashmap_begin] =
              prefix_tensor[k + scan_count + prefix_tensor_begin];
          unique_tensor[prefix_tensor[k + scan_count + prefix_tensor_begin] +
                        unique_tensor_begin] = batch_tensors[j][k + data_begin];
        }
      }
      scan_count += data_end - data_begin;
    }
  }
}

template <typename IdType>
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor>
BatchUnique(thrust::device_vector<IdType*> batch_tensors,
            thrust::device_vector<IdType*> segments_ptr, int64_t num_segments,
            int64_t num_batchs, torch::ScalarType IdTypeClass) {
  // create hashmaps
  torch::Tensor hashmap_ptr = torch::empty(
      num_batchs + 1, torch::dtype(IdTypeClass).device(torch::kCUDA));
  torch::Tensor item_prefix_ptr = torch::empty(
      num_batchs + 1, torch::dtype(IdTypeClass).device(torch::kCUDA));

  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      it(0), it(num_batchs + 1),
      [segments_data = segments_ptr.data(), num_segments, num_batchs,
       out_hashmap = hashmap_ptr.data_ptr<IdType>(),
       out_item_prefix =
           item_prefix_ptr.data_ptr<IdType>()] __device__(int64_t i) mutable {
        int64_t total_count = 0;
        int64_t batch_count = 0;
        for (int64_t k = 0; k < num_segments; k++) {
          total_count += segments_data[k][i];
          if (i < num_batchs) {
            batch_count += (segments_data[k][i + 1] - segments_data[k][i]);
          }
        }
        out_item_prefix[i] = total_count + i;
        if (i < num_batchs) {
          out_hashmap[i] =
              2 * (1 << static_cast<uint32_t>(log2(batch_count) + 1));
        }
      });

  thrust::device_ptr<IdType> wrapper_hashmap_ptr(
      static_cast<IdType*>(hashmap_ptr.data_ptr<IdType>()));
  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(wrapper_hashmap_ptr),
                           num_batchs + 1);
  thrust::device_ptr<IdType> wrapper_item_prefix_ptr(
      static_cast<IdType*>(item_prefix_ptr.data_ptr<IdType>()));
  IdType dir_size = wrapper_hashmap_ptr[num_batchs];
  IdType total_size = wrapper_item_prefix_ptr[num_batchs];

  IdType MAX = std::numeric_limits<IdType>::max();
  torch::Tensor key_tensor =
      torch::full(dir_size, -1, torch::dtype(IdTypeClass).device(torch::kCUDA));
  torch::Tensor index_tensor = torch::full(
      dir_size, MAX, torch::dtype(IdTypeClass).device(torch::kCUDA));
  torch::Tensor item_prefix_tensor =
      torch::empty(total_size, torch::dtype(IdTypeClass).device(torch::kCUDA));
  torch::Tensor unique_tensor_ptr = torch::empty(
      num_batchs + 1, torch::dtype(IdTypeClass).device(torch::kCUDA));

  constexpr int BLOCK_SIZE = 256;
  int TILE_SIZE = 8;
  int blocks = (num_batchs + TILE_SIZE - 1) / TILE_SIZE;

  _InsertHashmaps<IdType, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), item_prefix_tensor.data_ptr<IdType>(),
      item_prefix_ptr.data_ptr<IdType>(), unique_tensor_ptr.data_ptr<IdType>(),
      thrust::raw_pointer_cast(batch_tensors.data()),
      thrust::raw_pointer_cast(segments_ptr.data()), num_batchs, num_segments);

  thrust::device_ptr<IdType> wrapper_unique_tensor_ptr(
      static_cast<IdType*>(unique_tensor_ptr.data_ptr<IdType>()));
  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(wrapper_unique_tensor_ptr),
                           num_batchs + 1);
  int64_t unique_tensor_size = wrapper_unique_tensor_ptr[num_batchs];
  torch::Tensor unique_tensor = torch::empty(
      unique_tensor_size, torch::dtype(IdTypeClass).device(torch::kCUDA));

  // update hashmaps and generate unique_tensor
  torch::Tensor value_tensor = torch::empty_like(index_tensor);

  _UpdateHashmaps<IdType, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      value_tensor.data_ptr<IdType>(), hashmap_ptr.data_ptr<IdType>(),
      item_prefix_tensor.data_ptr<IdType>(), item_prefix_ptr.data_ptr<IdType>(),
      unique_tensor.data_ptr<IdType>(), unique_tensor_ptr.data_ptr<IdType>(),
      thrust::raw_pointer_cast(batch_tensors.data()),
      thrust::raw_pointer_cast(segments_ptr.data()), num_batchs, num_segments);

  return {unique_tensor, unique_tensor_ptr, key_tensor, value_tensor,
          hashmap_ptr};
}

std::tuple<torch::Tensor, torch::Tensor> BatchUniqueCUDA(
    std::vector<torch::Tensor> batch_tensors,
    std::vector<torch::Tensor> segment_ptrs, int64_t num_batchs) {
  int num_segments = batch_tensors.size();
  thrust::host_vector<int64_t*> h_batch_tensors_ptr(num_segments);
  thrust::host_vector<int64_t*> h_segment_ptrs_ptrs(num_segments);

  for (int i = 0; i < num_segments; i++) {
    h_batch_tensors_ptr[i] = batch_tensors[i].data_ptr<int64_t>();
    h_segment_ptrs_ptrs[i] = segment_ptrs[i].data_ptr<int64_t>();
  }

  thrust::device_vector<int64_t*> d_batch_tensors_ptr = h_batch_tensors_ptr;
  thrust::device_vector<int64_t*> d_segment_ptrs_ptrs = h_segment_ptrs_ptrs;

  torch::Tensor unique_tensor, unique_tensor_ptr, key_tensor, value_tensor,
      hashmap_ptr;
  std::tie(unique_tensor, unique_tensor_ptr, key_tensor, value_tensor,
           hashmap_ptr) =
      BatchUnique<int64_t>(d_batch_tensors_ptr, d_segment_ptrs_ptrs,
                           num_segments, num_batchs, torch::kInt64);
  return {unique_tensor, unique_tensor_ptr};
}

/////////////////////////////////// BatchReindex  /////////////////////////

template <typename IdType, int BLOCK_SIZE>
__global__ void _SearchHashmaps(IdType* key_tensor, IdType* value_tensor,
                                IdType* hashmap_ptr, IdType** batch_tensor,
                                IdType** reindex_batch_tensor,
                                IdType** segments_ptr, int64_t num_batchs,
                                int64_t num_segments) {
  int tid = threadIdx.x;
  int thread_stride = blockDim.x;
  assert(thread_stride == BLOCK_SIZE);

  int bid = blockIdx.x;
  int block_stride = gridDim.x;

  for (int i = bid; i < num_batchs; i += block_stride) {
    int64_t hashmap_begin = hashmap_ptr[i];
    int64_t dir_size = hashmap_ptr[i + 1] - hashmap_begin;
    RelabelHashmap<IdType> table(key_tensor + hashmap_begin,
                                 value_tensor + hashmap_begin, dir_size);

    for (int j = 0; j < num_segments; j++) {
      int64_t data_begin = segments_ptr[j][i];
      int64_t data_end = segments_ptr[j][i + 1];
      for (int k = tid; k < data_end - data_begin; k += thread_stride) {
        reindex_batch_tensor[j][k + data_begin] =
            table.SearchForValue(batch_tensor[j][k + data_begin]);
      }
    }
  }
}

template <typename IdType>
void BatchReindex(thrust::device_vector<IdType*> batch_tensors,
                  thrust::device_vector<IdType*> reindex_batch_tensors,
                  thrust::device_vector<IdType*> segments_ptr,
                  torch::Tensor key_tensor, torch::Tensor value_tensor,
                  torch::Tensor hashmap_ptr, int64_t num_segments,
                  int64_t num_batchs) {
  constexpr int BLOCK_SIZE = 256;
  int TILE_SIZE = 8;
  int blocks = (num_batchs + TILE_SIZE - 1) / TILE_SIZE;
  _SearchHashmaps<IdType, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
      key_tensor.data_ptr<IdType>(), value_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(),
      thrust::raw_pointer_cast(batch_tensors.data()),
      thrust::raw_pointer_cast(reindex_batch_tensors.data()),
      thrust::raw_pointer_cast(segments_ptr.data()), num_batchs, num_segments);
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
BatchRelabelCUDA(std::vector<torch::Tensor> batch_tensors,
                 std::vector<torch::Tensor> segment_ptrs, int64_t num_batchs) {
  int num_segments = batch_tensors.size();

  std::vector<torch::Tensor> relabel_batch_tensor;
  thrust::host_vector<int64_t*> h_batch_tensors_ptr(num_segments);
  thrust::host_vector<int64_t*> h_segment_ptrs_ptrs(num_segments);
  thrust::host_vector<int64_t*> h_reindex_batch_tensors_ptr(num_segments);

  for (int i = 0; i < num_segments; i++) {
    h_batch_tensors_ptr[i] = batch_tensors[i].data_ptr<int64_t>();
    h_segment_ptrs_ptrs[i] = segment_ptrs[i].data_ptr<int64_t>();

    relabel_batch_tensor.push_back(torch::empty_like(batch_tensors[i]));
    h_reindex_batch_tensors_ptr[i] =
        relabel_batch_tensor[i].data_ptr<int64_t>();
  }

  thrust::device_vector<int64_t*> d_batch_tensors_ptr = h_batch_tensors_ptr;
  thrust::device_vector<int64_t*> d_segment_ptrs_ptrs = h_segment_ptrs_ptrs;
  thrust::device_vector<int64_t*> d_reindex_batch_tensors_ptr =
      h_reindex_batch_tensors_ptr;

  torch::Tensor unique_tensor, unique_tensor_ptr, key_tensor, value_tensor,
      hashmap_ptr;
  std::tie(unique_tensor, unique_tensor_ptr, key_tensor, value_tensor,
           hashmap_ptr) =
      BatchUnique<int64_t>(d_batch_tensors_ptr, d_segment_ptrs_ptrs,
                           num_segments, num_batchs, torch::kInt64);

  BatchReindex<int64_t>(d_batch_tensors_ptr, d_reindex_batch_tensors_ptr,
                        d_segment_ptrs_ptrs, key_tensor, value_tensor,
                        hashmap_ptr, num_segments, num_batchs);

  return {unique_tensor, unique_tensor_ptr, relabel_batch_tensor, segment_ptrs};
}

}  // namespace impl
}  // namespace gs