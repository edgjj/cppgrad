// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/cuda_executor.hpp"
#include "cppgrad/exceptions/out_of_memory.hpp"
#include <cub/util_allocator.cuh>

namespace cppgrad {

// this is all for single device; to be changed

struct CUDAAllocatorWrapper {

    static cub::CachingDeviceAllocator& get()
    {
        // indirection to make init routine called once
        static CUDAAllocatorWrapper wrapper;
        return wrapper._allocator;
    }

private:
    CUDAAllocatorWrapper()
        : _allocator(9, // bin growth
            3, // min bins
            9, // max bin
            1 << 28) // max cached bytes
    {
    }

    CUDAAllocatorWrapper(const CUDAAllocatorWrapper&) = delete;
    CUDAAllocatorWrapper(CUDAAllocatorWrapper&&) = delete;

    cub::CachingDeviceAllocator _allocator;
};

std::byte* CUDA::allocate(std::size_t count)
{
    std::byte* ptr;

    // auto result = cudaMalloc(&ptr, count);
    auto result = CUDAAllocatorWrapper::get().DeviceAllocate((void**)(&ptr), count);

    if (result != cudaSuccess) {
        // retrieve available mem to print out later
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        throw exceptions::OutOfMemoryError("cpu", count, free_mem);
    }

    return ptr;
}

void CUDA::deallocate(std::byte* ptr)
{
    CUDAAllocatorWrapper::get().DeviceFree((void*)ptr);
}

impl::Executor& CUDA::get_executor()
{
    // dispatch between AVX/SSE/etc executors there?
    static impl::CUDAExecutor executor(*this);

    return executor;
}

int CUDA::num_devices()
{
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);

    return nDevices;
}

bool CUDA::clear_cache()
{
    return CUDAAllocatorWrapper::get().FreeAllCached() == cudaSuccess;
}

void CUDA::sync()
{
    cudaDeviceSynchronize();
}

}