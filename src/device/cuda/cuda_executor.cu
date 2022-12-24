#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/cuda/cuda_executor.hpp"
#include "cppgrad/device/cuda/kernels/ops_kernels.cuh"
#include "cppgrad/tensor/ops/op_wrapper.hpp"

#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::impl {

CUDAExecutor::CUDAExecutor(CUDA& parent_device)
    : _parent(parent_device)
{
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
    auto fn = [value](auto out, auto p1, auto p2) {
        using Type = typename decltype(out)::Type;

        auto fill_value = *reinterpret_cast<Type*>(value);

        CPPGRAD_CUDA_LAUNCH(fill_kernel, out.size())
        (out, fill_value);
    };

    for_each_type(OpWrapper1D { std::move(fn), tensor, tensor, tensor }, tensor.dtype());
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

void CUDAExecutor::div(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(div_kernel, out.size())
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
    auto fn = [](auto out, auto p1, auto p2) {
        using Type = typename decltype(out)::Type;

        CPPGRAD_CUDA_LAUNCH(dot_kernel, p1.size())
        (p1, p2, out);
    };

    // we'll use that in incorrect way
    cudaMemset(dst.data(), 0, dst.nbytes());

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CUDAExecutor::matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [&](auto out, auto p1, auto p2) {
        // use p1 rows as dimX & p2 cols as dimY; result shape is {p1_rows, p2_cols}
        CPPGRAD_CUDA_LAUNCH_2D(matmul_kernel, out.size(0), out.size(1)) // was p2.size(1), p2.size(0)
        (p1, p2, out);
    };

    for_each_type(OpWrapper2D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CUDAExecutor::log(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(log_kernel, out.size())
        (p1, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CUDAExecutor::exp(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(exp_kernel, out.size())
        (p1, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CUDAExecutor::relu(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(relu_kernel, out.size())
        (p1, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CUDAExecutor::tanh(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(tanh_kernel, out.size())
        (p1, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CUDAExecutor::sign(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(sign_kernel, out.size())
        (p1, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CUDAExecutor::neg(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(neg_kernel, out.size())
        (p1, out);
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CUDAExecutor::cmp(const Tensor& lhs, const Tensor& rhs, Tensor& dst, CompareType cmp_type)
{
}

}