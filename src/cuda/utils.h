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

template <typename IdType, typename DType>
void cub_sortPairs(IdType* input_key, DType* input_value, IdType* output_key,
                   DType* output_value, int num_items) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_key,
                                  output_key, input_value, output_value,
                                  num_items);

  c10::Allocator* cuda_allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
  d_temp_storage = _temp_data.get();

  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, input_key,
                                  output_key, input_value, output_value,
                                  num_items);
}

}  // namespace impl
}  // namespace gs

#endif