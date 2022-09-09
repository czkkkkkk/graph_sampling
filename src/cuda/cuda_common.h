#ifndef GS_CUDA_CUDA_COMMON_H_
#define GS_CUDA_CUDA_COMMON_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <torch/torch.h>
#include <cub/cub.cuh>

#define WARP_SIZE 32
#define MIN(x, y) ((x < y) ? x : y)

#endif