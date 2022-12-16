#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/cuda_executor.hpp"
#include "cppgrad/exceptions/out_of_memory.hpp"

namespace cppgrad {

std::byte* CUDA::allocate(std::size_t count, std::align_val_t alignment)
{
    std::byte* ptr;

    auto result = cudaMalloc(&ptr, count);

    if (result != cudaSuccess) {
        // retrieve available mem to print out later
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        throw exceptions::OutOfMemoryError(type(), count, free_mem);
    }

    return ptr;
}

void CUDA::deallocate(std::byte* ptr, std::align_val_t alignment)
{
    cudaDeviceSynchronize();
    cudaFree(ptr);
}

impl::Executor& CUDA::get_executor()
{
    // dispatch between AVX/SSE/etc executors there?
    static impl::CUDAExecutor executor;

    return executor;
}

Device* CUDA::clone()
{
    return new CUDA();
}

std::string_view CUDA::type()
{
    return "cuda"; // include device number there later like cuda:X
}

int CUDA::num_devices()
{
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);

    return nDevices;
}

}