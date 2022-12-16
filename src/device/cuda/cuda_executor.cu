#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/cuda_executor.hpp"

#include "cppgrad/device/cuda/kernels/ops_kernels.cuh"
#include "cppgrad/device/cuda/kernels/strided_copy_kernel.cuh"

#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::impl {

void CUDAExecutor::copy(const std::byte* from, std::byte* to, std::size_t count, CopyType copy_type)
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
    auto fn = [&](auto tag) {
        using Type = decltype(tag);

        auto v = reinterpret_cast<Type*>(value);
        auto out = reinterpret_cast<Type*>(tensor.data());

        CPPGRAD_CUDA_LAUNCH(impl::fill_kernel, tensor.numel())
        (out, tensor.numel(), *v);
    };

    for_each_type(std::move(fn), tensor.dtype());
}

void CUDAExecutor::sum(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [&](auto tag) {
        using Type = decltype(tag);
        auto p1 = reinterpret_cast<const Type*>(lhs.data());
        auto p2 = reinterpret_cast<const Type*>(rhs.data());

        auto out = reinterpret_cast<Type*>(dst.data());

        CPPGRAD_CUDA_LAUNCH(sum_kernel, dst.numel())
        (p1, p2, out, dst.numel());
    };

    for_each_type(std::move(fn), dst.dtype());
}

void CUDAExecutor::sub(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [&](auto tag) {
        using Type = decltype(tag);
        auto p1 = reinterpret_cast<const Type*>(lhs.data());
        auto p2 = reinterpret_cast<const Type*>(rhs.data());

        auto out = reinterpret_cast<Type*>(dst.data());

        CPPGRAD_CUDA_LAUNCH(sub_kernel, dst.numel())
        (p1, p2, out, dst.numel());
    };

    for_each_type(std::move(fn), dst.dtype());
}

void CUDAExecutor::mul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [&](auto tag) {
        using Type = decltype(tag);
        auto p1 = reinterpret_cast<const Type*>(lhs.data());
        auto p2 = reinterpret_cast<const Type*>(rhs.data());

        auto out = reinterpret_cast<Type*>(dst.data());

        CPPGRAD_CUDA_LAUNCH(mul_kernel, dst.numel())
        (p1, p2, out, dst.numel());
    };

    for_each_type(std::move(fn), dst.dtype());
}

void CUDAExecutor::pow(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    // matrix mul
    auto fn = [&](auto tag) {
        using Type = decltype(tag);
        auto p1 = reinterpret_cast<const Type*>(lhs.data());
        auto p2 = reinterpret_cast<const Type*>(rhs.data());

        auto out = reinterpret_cast<Type*>(dst.data());

        CPPGRAD_CUDA_LAUNCH(pow_kernel, dst.numel())
        (p1, p2, out, dst.numel());
    };

    for_each_type(std::move(fn), dst.dtype());
}

void CUDAExecutor::dot(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    // matrix mul
    auto fn = [&](auto tag) {
        using Type = decltype(tag);
        auto p1 = reinterpret_cast<const Type*>(lhs.data());
        auto p2 = reinterpret_cast<const Type*>(rhs.data());

        auto out = reinterpret_cast<Type*>(dst.data());

        CPPGRAD_CUDA_LAUNCH(dot_kernel, dst.numel())
        (p1, p2, out, dst.numel());
    };

    for_each_type(std::move(fn), dst.dtype());
}

void CUDAExecutor::matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [&](auto tag) {
        using Type = decltype(tag);
        auto p1 = reinterpret_cast<const Type*>(lhs.data());
        auto p2 = reinterpret_cast<const Type*>(rhs.data());

        auto out = reinterpret_cast<Type*>(dst.data());

        CPPGRAD_CUDA_LAUNCH(matmul_kernel, dst.numel())
        (p1, p2, out, dst.numel());
    };

    for_each_type(std::move(fn), dst.dtype());
}

void CUDAExecutor::relu(const Tensor& lhs, Tensor& dst)
{
}

void CUDAExecutor::tanh(const Tensor& lhs, Tensor& dst)
{
}

void CUDAExecutor::cmp(const Tensor& lhs, const Tensor& rhs, Tensor& dst, CompareType cmp_type)
{
}

}