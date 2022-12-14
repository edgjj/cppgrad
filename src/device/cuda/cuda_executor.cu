#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/cuda_executor.hpp"
#include "cppgrad/device/cuda/kernels/fill_kernel.cuh"
#include "cppgrad/device/cuda/kernels/strided_copy_kernel.cuh"
#include "cppgrad/tensor/tensor.hpp"

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

void CUDAExecutor::strided_copy(const Tensor& from, Tensor& to)
{
    FOREACH_TYPE(from.dtype(),
        impl::strided_copy_impl,
        from.data(),
        to.data(),
        from.shape().data(),
        from.strides().data(),
        to.strides().data(),
        from.shape().size());
}

void CUDAExecutor::fill(Tensor& tensor, std::byte* value)
{
    FOREACH_TYPE(tensor.dtype(),
        impl::fill_impl,
        tensor.data(),
        value,
        tensor.numel());
}

}