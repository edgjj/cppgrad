#ifndef CPPGRAD_CUDA_OPS_KERNELS_CUH
#define CPPGRAD_CUDA_OPS_KERNELS_CUH

#include "cppgrad/device/cuda/cuda_defs.hpp"
#include "cppgrad/tensor/util/strided_span.hpp"

namespace cppgrad::impl {

template <typename T>
__global__ void strided_copy_kernel(ConstStridedSpan<T> p1, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        out[i] = p1[i];
    }
}

template <typename T>
__global__ void fill_kernel(T* data, size_t size, T val)
{
    CPPGRAD_CUDA_1D_LOOP(i, size)
    {
        data[i] = val;
    }
}

template <typename T>
__global__ void sum_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        out[i] = p1[i] + p2[i];
    }
}

template <typename T>
__global__ void sub_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        out[i] = p1[i] - p2[i];
    }
}

template <typename T>
__global__ void mul_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        out[i] = p1[i] * p2[i];
    }
}

template <typename T>
__global__ void pow_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        // escape to global namespace to avoid cppgrad::pow
        out[i] = ::pow(p1[i], p2[i]);
    }
}

template <typename T>
__global__ void dot_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out)
{
    out[0] = T(0);
    // incorrect due to reduction
    CPPGRAD_CUDA_1D_LOOP(i, p1.size())
    {
        out[0] += p1[i] * p2[i];
    }
}

template <typename T>
__global__ void matmul_kernel(ConstStridedSpan2D<T> p1, ConstStridedSpan2D<T> p2, StridedSpan2D<T> out)
{
}

}

#endif