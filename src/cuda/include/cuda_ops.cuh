#pragma once

#include <torch/script.h>
#include <ATen/Context.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include "ops.h"
#include "cuda_allocate.cuh"

#include <cub/cub.cuh>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>


#define FULL_MASK 0xffffffff
#define WARP_SIZE 32


// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
#define CHECK_SAME_TYPE(x,y)                                                   \
  AT_ASSERTM(x.dtype() == y.dtype(), #x #y " must be same dtype")

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    cudaError_t result = (x);                                                  \
    if (result != cudaSuccess) {                                               \
      const char *msg = cudaGetErrorString(result);                            \
      std::stringstream safe_call_ss;                                          \
      safe_call_ss << "\nerror: " #x " failed with error"                      \
                   << "\nfile: " << __FILE__ << "\nline: " << __LINE__         \
                   << "\nmsg: " << msg;                                        \
      throw std::runtime_error(safe_call_ss.str());                            \
    }                                                                          \
  } while (0)


#define CUSPARSE_CALL(func)                                        \
  {                                                                \
    cusparseStatus_t e = (func);                                   \
    CHECK(e == CUSPARSE_STATUS_SUCCESS)                            \
        << "CUSPARSE ERROR: " << e;                                \
  }

#define MIN(x, y) ((x < y) ? x : y)
#define MAX(x, y) ((x > y) ? x : y)

// wrapper
template<typename IdType>
inline void cub_exclusiveSum(
    IdType* arrays,
    const IdType array_length
){
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        arrays,
        arrays,
        array_length
    );

    c10::Allocator* cuda_allocator =  ogs::get_cuda_allocator();
    c10::DataPtr _temp_data = cuda_allocator->allocate(temp_storage_bytes);
    d_temp_storage = _temp_data.get();

    CUDA_SAFE_CALL(cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        arrays,
        arrays,
        array_length
    ));
}

__device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

torch::Tensor unique_single(
  torch::Tensor data
);

std::tuple<torch::Tensor, torch::Tensor> relabel_single(
  torch::Tensor data
);

std::tuple<torch::Tensor, std::vector<torch::Tensor>> relabel(
  std::vector<torch::Tensor> data
);