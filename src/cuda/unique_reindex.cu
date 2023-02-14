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

// Relabel
template <typename IdType, bool need_cached>
inline std::vector<torch::Tensor> Unique(torch::Tensor total_tensor) {
  at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t stream = torch_stream.stream();

  int num_items = total_tensor.numel();
  int dir_size = UpPower(num_items);

  IdType MAX = std::numeric_limits<IdType>::max();
  torch::Tensor key_tensor = torch::full(dir_size, -1, total_tensor.options());
  torch::Tensor index_tensor =
      torch::full(dir_size, MAX, total_tensor.options());

  // insert
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device.on(stream), it(0), it(num_items),
                   [key = key_tensor.data_ptr<IdType>(),
                    index = index_tensor.data_ptr<IdType>(),
                    in = total_tensor.data_ptr<IdType>(), num_items,
                    dir_size] __device__(IdType i) mutable {
                     RelabelHashmap<IdType> table(key, index, dir_size);
                     table.Update(in[i], i);
                   });

  // prefix sum
  torch::Tensor item_prefix_tensor =
      torch::empty(num_items + 1, total_tensor.options());
  IdType* prefix_dptr = item_prefix_tensor.data_ptr<IdType>();
  thrust::for_each(thrust::device.on(stream), it(0), it(num_items),
                   [key = key_tensor.data_ptr<IdType>(),
                    index = index_tensor.data_ptr<IdType>(),
                    in = total_tensor.data_ptr<IdType>(), count = prefix_dptr,
                    num_items, dir_size] __device__(IdType i) mutable {
                     RelabelHashmap<IdType> table(key, index, dir_size);
                     count[i] = table.SearchForValue(in[i]) == i ? 1 : 0;
                   });
  cub_exclusiveSum<IdType>(prefix_dptr, num_items + 1, stream);

  // unique
  // cudaStreamSynchronize(stream);
  // thrust::device_ptr<IdType> item_prefix(static_cast<IdType*>(prefix_dptr));
  // int64_t tot = item_prefix[num_items];
  int64_t tot;
  CUDA_CALL((cudaMemcpyAsync(&tot, prefix_dptr + num_items, sizeof(IdType),
                             cudaMemcpyDeviceToHost, stream)));
  CUDA_CALL((cudaStreamSynchronize(stream)));
  torch::Tensor unique_tensor = torch::empty(tot, total_tensor.options());

  torch::Tensor value_tensor;
  if (need_cached) {
    value_tensor = torch::full(dir_size, -1, total_tensor.options());
  }

  thrust::for_each(
      thrust::device.on(stream), it(0), it(num_items),
      [key = key_tensor.data_ptr<IdType>(),
       index = index_tensor.data_ptr<IdType>(),
       in = total_tensor.data_ptr<IdType>(), prefix = prefix_dptr,
       unique = unique_tensor.data_ptr<IdType>(),
       cache_value = need_cached ? value_tensor.data_ptr<IdType>() : nullptr,
       num_items, dir_size] __device__(IdType i) mutable {
        RelabelHashmap<IdType> table(key, index, dir_size);
        IdType pos = table.SearchForPos(in[i]);
        if (index[pos] == i) {
          unique[prefix[i]] = in[i];
          if (cache_value) {
            cache_value[pos] = prefix[i];
          }
        }
      });

  if (need_cached) {
    return {unique_tensor, key_tensor, value_tensor};
  } else {
    return {unique_tensor};
  }
}

template <typename IdType>
inline torch::Tensor Relabel(torch::Tensor total_tensor,
                             torch::Tensor key_tensor,
                             torch::Tensor value_tensor) {
  at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t stream = torch_stream.stream();

  int num_items = total_tensor.numel();
  torch::Tensor relabel_tensor = torch::zeros_like(total_tensor);
  int dir_size = key_tensor.numel();

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device.on(stream), it(0), it(num_items),
                   [key = key_tensor.data_ptr<IdType>(),
                    value = value_tensor.data_ptr<IdType>(),
                    in = total_tensor.data_ptr<IdType>(),
                    out = relabel_tensor.data_ptr<IdType>(),
                    dir_size] __device__(IdType i) mutable {
                     RelabelHashmap<IdType> table(key, value, dir_size);
                     out[i] = table.SearchForValue(in[i]);
                   });
  return relabel_tensor;
}

torch::Tensor TensorUniqueCUDA(torch::Tensor node_ids) {
  torch::Tensor ret_tensor = Unique<int64_t, false>(node_ids)[0];
  return ret_tensor;
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> RelabelCUDA(
    std::vector<torch::Tensor> mapping_tensor,
    std::vector<torch::Tensor> data_requiring_relabel) {
  std::vector<int64_t> split_sizes;
  for (auto d : data_requiring_relabel) {
    split_sizes.push_back(d.numel());
  }

  torch::Tensor total_tensor = torch::cat(data_requiring_relabel, 0);

  std::vector<torch::Tensor> unique_result =
      Unique<int64_t, true>(torch::cat(mapping_tensor, 0));
  torch::Tensor unique_tensor = unique_result[0];
  torch::Tensor reindex_tensor =
      Relabel<int64_t>(total_tensor, unique_result[1], unique_result[2]);

  std::vector<torch::Tensor> ret =
      reindex_tensor.split_with_sizes(split_sizes, 0);

  return std::make_tuple(unique_tensor, ret);
}
}  // namespace impl

}  // namespace gs