#ifndef GS_CUDA_UTILS_H_
#define GS_CUDA_UTILS_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>
#include "cuda_common.h"

namespace gs {
namespace impl {

template <typename IdType>
void cub_exclusiveSum(IdType* arrays, const IdType array_length) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);

  c10::Allocator* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
  d_temp_storage = _temp_data.get();
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);
}

template <typename IdType>
void cub_inclusiveSum(IdType* arrays, int32_t array_length) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);

  c10::Allocator* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
  d_temp_storage = _temp_data.get();
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);
}

template <typename KeyType, typename ValueType>
void cub_sortPairsDescending(cub::DoubleBuffer<KeyType> d_keys,
                             cub::DoubleBuffer<ValueType> d_values,
                             int32_t num_items) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys, d_values, num_items);
  // Allocate temporary storage
  c10::Allocator* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
  d_temp_storage = _temp_data.get();

  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
                                            d_keys, d_values, num_items);
}

}  // namespace impl
}  // namespace gs

#endif