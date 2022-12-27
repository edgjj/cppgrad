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

void CUDAExecutor::random_fill(Tensor& tensor, double lower_bound, double upper_bound)
{
    auto fn = [lower_bound, upper_bound](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(random_fill_kernel, out.size())
        (out, lower_bound, upper_bound);
    };

    for_each_type(OpWrapper1D { std::move(fn), tensor, tensor, tensor }, tensor.dtype());
}

void CUDAExecutor::add(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(add_kernel, out.size())
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
        CPPGRAD_CUDA_LAUNCH(reduce_kernel, p1.size())
        (p1, p2, out, DotReduceTag {});
    };

    // we'll use that in incorrect way
    cudaMemset(dst.data(), 0, dst.nbytes());

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CUDAExecutor::sum(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        CPPGRAD_CUDA_LAUNCH(reduce_kernel, p1.size())
        (p1, p2, out, SumReduceTag {});
    };

    // we'll use that in incorrect way
    cudaMemset(dst.data(), 0, dst.nbytes());

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CUDAExecutor::matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [&](auto out, auto p1, auto p2) {
        auto n_col_rows = p1.size(1); // or p2.size(0)
        if (n_col_rows <= 64) {
            matmul_kernel<typename decltype(out)::Type, 8>
                <<<impl::grid_size_for_N_2D_mm(out.size(1), out.size(0), 8), dim3(8, 8)>>>(p1, p2, out);
        } else if (n_col_rows <= 256) {
            matmul_kernel<typename decltype(out)::Type, 16>
                <<<impl::grid_size_for_N_2D_mm(out.size(1), out.size(0), 16), dim3(16, 16)>>>(p1, p2, out);
        } else {
            matmul_kernel<typename decltype(out)::Type, 32>
                <<<impl::grid_size_for_N_2D_mm(out.size(1), out.size(0), 32), dim3(32, 32)>>>(p1, p2, out);
        }
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