#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/fill_kernel.cuh"

namespace cppgrad {

std::byte* CUDA::allocate(std::size_t count, std::align_val_t alignment, std::string& err)
{
    std::byte* ptr;

    auto result = cudaMalloc(&ptr, count);
    if (result != cudaSuccess) {
        // retrieve available mem to print out later
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        err += "[ ";
        err += type();
        err += " ]";
        err += "Device out of memory. Tried to allocate: ";
        err += std::to_string(count);
        err += " bytes. ";
        err += "Available memory: ";
        err += std::to_string(free_mem);
        err += " bytes.";
        return nullptr;
    }

    return ptr;
}

void CUDA::deallocate(std::byte* ptr, std::align_val_t alignment)
{
    cudaDeviceSynchronize();
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

void CUDA::assign(std::byte* pos, std::byte* value, DType type, std::size_t count)
{
    copy(value, pos, type_size(type) * count);
}

void CUDA::fill(std::byte* pos, std::byte* value, DType type, std::size_t count)
{
    FOREACH_TYPE(type, impl::fill_impl, pos, value, count);
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