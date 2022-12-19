#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/cuda_executor.hpp"
#include "cppgrad/device/cuda/kernels/ops_kernels.cuh"
#include "cppgrad/tensor/ops/op_wrapper.hpp"

#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::impl {

CUDAExecutor::CUDAExecutor(CUDA& parent_device)
{
    // pre-allocate memory for reduce ops
    // kernels should care of proper memory initialization.
    _reduce_mem = parent_device.allocate(CPPGRAD_CUDA_MAX_GRID_SIZE * 16, std::align_val_t(0));
}

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
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(strided_copy_kernel, out.size())
        (p1, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), to, from, from }, to.dtype());
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
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(sum_kernel, out.size())
        (p1, p2, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CUDAExecutor::sub(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(sub_kernel, out.size())
        (p1, p2, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CUDAExecutor::mul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(mul_kernel, out.size())
        (p1, p2, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CUDAExecutor::pow(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(pow_kernel, out.size())
        (p1, p2, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CUDAExecutor::dot(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    // matrix mul
    auto fn = [blk_results = _reduce_mem](auto out, auto p1, auto p2) {
        using Type = typename decltype(out)::Type;

        CPPGRAD_CUDA_LAUNCH(dot_kernel, p1.size())
        (p1, p2, out, reinterpret_cast<Type*>(blk_results));
    };

    // we'll use that in incorrect way
    cudaMemset(dst.data(), 0, dst.nbytes());

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CUDAExecutor::matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    // auto fn = [&](auto tag) {
    //     using Type = decltype(tag);
    //     auto p1 = reinterpret_cast<const Type*>(lhs.data());
    //     auto p2 = reinterpret_cast<const Type*>(rhs.data());

    //     auto out = reinterpret_cast<Type*>(dst.data());

    //     CPPGRAD_CUDA_LAUNCH(matmul_kernel, dst.numel())
    //     (p1, p2, out, dst.numel());
    // };

    // for_each_type(std::move(fn), dst.dtype());

    auto fn = [&](auto out, auto p1, auto p2) {
        // CPPGRAD_CUDA_LAUNCH(matmul_kernel, dst.numel())
        // (p1, p2, out);
    };

    // we'll use that first in incorrect way
    cudaMemset(dst.data(), 0, dst.nbytes());

    for_each_type(OpWrapper2D { std::move(fn), dst, lhs, rhs }, dst.dtype());
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