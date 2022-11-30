#include "cppgrad/device/cuda/cuda.hpp"

namespace cppgrad {

std::byte* CUDA::allocate(std::size_t count, std::align_val_t alignment)
{
    std::byte* ptr;
    cudaMalloc(&ptr, count);
    return ptr;
}

void CUDA::deallocate(std::byte* ptr, std::align_val_t alignment)
{
    cudaFree(ptr);
}

void CUDA::copy(std::byte* from, std::byte* to, std::size_t count)
{
    cudaMemcpy(to, from, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void CUDA::copy_from_host(std::byte* from, std::byte* to, std::size_t count)
{
    cudaMemcpy(to, from, count, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

void CUDA::copy_to_host(std::byte* from, std::byte* to, std::size_t count)
{
    cudaMemcpy(to, from, count, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

int CUDA::num_devices()
{
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);

    return nDevices;
}

}