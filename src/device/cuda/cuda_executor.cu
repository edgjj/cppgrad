#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/cuda_executor.hpp"
#include "cppgrad/device/cuda/fill_kernel.cuh"
#include "cppgrad/device/cuda/strided_copy_kernel.cuh"

namespace cppgrad::impl {

void CUDAExecutor::copy(std::byte* from, std::byte* to, std::size_t count, CopyType copy_type)
{
    cudaMemcpyKind kind;

    switch (copy_type) {
    case Homogeneous:
        kind = cudaMemcpyDeviceToDevice;
        break;
    case DeviceToHost:
        kind = cudaMemcpyDeviceToHost;
        break;
    case HostToDevice:
        kind = cudaMemcpyHostToDevice;
        break;
    }

    cudaMemcpy(to, from, count, kind);
}

void CUDAExecutor::strided_copy(std::byte* from,
    std::byte* to,
    DType type,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& from_strides,
    const std::vector<size_t>& to_strides)
{
    FOREACH_TYPE(type, impl::strided_copy_impl, from, to, shape.data(), from_strides.data(), to_strides.data(), shape.size());
}

void CUDAExecutor::fill(std::byte* pos, std::byte* value, DType type, std::size_t count)
{
    FOREACH_TYPE(type, impl::fill_impl, pos, value, count);
}

}