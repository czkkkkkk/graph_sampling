#pragma once
#include <c10/core/Allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace ogs{
    static c10::Allocator* get_cuda_allocator(){
        return c10::cuda::CUDACachingAllocator::get();
    }
};