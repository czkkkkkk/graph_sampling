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

}  // namespace impl
}  // namespace gs

#endif