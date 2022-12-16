#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/cuda_executor.hpp"
#include "cppgrad/device/cuda/kernels/fill_kernel.cuh"
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
    FOREACH_TYPE(tensor.dtype(),
        impl::fill_impl,
        tensor.data(),
        value,
        tensor.numel());
}

void CUDAExecutor::sum(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [&](auto tag) {
        using Type = decltype(tag);
        auto p1 = reinterpret_cast<const Type*>(lhs.data());
        auto p2 = reinterpret_cast<const Type*>(rhs.data());

        auto out = reinterpret_cast<Type*>(dst.data());
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
    };

    for_each_type(std::move(fn), dst.dtype());
}

void CUDAExecutor::matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    // matrix mul
    if (dst.shape().size() == 2) {
        auto fn = [&](auto tag) {
            using Type = decltype(tag);
            auto p1 = reinterpret_cast<const Type*>(lhs.data());
            auto p2 = reinterpret_cast<const Type*>(rhs.data());

            auto out = reinterpret_cast<Type*>(dst.data());
        };
        for_each_type(std::move(fn), dst.dtype());
    } else if (dst.shape().size() == 1) { // dot product
        auto fn = [&](auto tag) {
            using Type = decltype(tag);
            auto p1 = reinterpret_cast<const Type*>(lhs.data());
            auto p2 = reinterpret_cast<const Type*>(rhs.data());

            auto out = reinterpret_cast<Type*>(dst.data());
        };
    }
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