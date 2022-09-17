#ifndef GS_CUDA_UTILS_H_
#define GS_CUDA_UTILS_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>
#include "cuda_common.h"

#define DATA_TYPE_SWITCH(val, DType, ...)                         \
  do {                                                            \
    if ((val) == torch::kInt32) {                                 \
      typedef int32_t DType;                                      \
      { __VA_ARGS__ }                                             \
    } else if ((val) == torch::kInt64) {                          \
      typedef int64_t DType;                                      \
      { __VA_ARGS__ }                                             \
    } else if ((val) == torch::kFloat32) {                        \
      typedef float DType;                                        \
      { __VA_ARGS__ }                                             \
    } else {                                                      \
      LOG(FATAL) << "Data can only be int32 or int64 or float32"; \
    }                                                             \
  } while (0);

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

template <typename KeyType, typename ValueType>
void cub_sortPairs(cub::DoubleBuffer<KeyType> d_keys,
                   cub::DoubleBuffer<ValueType> d_values, int32_t num_items) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys,
                                  d_values, num_items);

  c10::Allocator* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
  d_temp_storage = _temp_data.get();

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys,
                                  d_values, num_items);
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

/*!
 * \brief Given a sorted array and a value this function returns the index
 * of the first element which compares greater than value.
 *
 * This function assumes 0-based index
 * @param A: ascending sorted array
 * @param n: size of the A
 * @param x: value to search in A
 * @return index, i, of the first element st. A[i]>x. If x>=A[n-1] returns n.
 * if x<A[0] then it returns 0.
 */
template <typename IdType>
__device__ IdType _UpperBound(const IdType* A, int64_t n, IdType x) {
  IdType l = 0, r = n, m = 0;
  while (l < r) {
    m = l + (r - l) / 2;
    if (A[m] <= x) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}

}  // namespace impl
}  // namespace gs

#endif