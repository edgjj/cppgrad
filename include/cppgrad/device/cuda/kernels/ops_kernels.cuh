#ifndef CPPGRAD_CUDA_OPS_KERNELS_CUH
#define CPPGRAD_CUDA_OPS_KERNELS_CUH

#include "cppgrad/device/cuda/cuda_defs.hpp"

namespace cppgrad::impl {

template <typename T>
__global__ void fill_kernel(T* data, size_t size, T val)
{
    CPPGRAD_CUDA_1D_LOOP(i, size)
    {
        data[i] = val;
    }
}

template <typename T>
__global__ void sum_kernel(const T* p1, const T* p2, T* out, size_t size)
{
    CPPGRAD_CUDA_1D_LOOP(i, size)
    {
        out[i] = p1[i] + p2[i];
    }
}

template <typename T>
__global__ void sub_kernel(const T* p1, const T* p2, T* out, size_t size)
{
    CPPGRAD_CUDA_1D_LOOP(i, size)
    {
        out[i] = p1[i] - p2[i];
    }
}

template <typename T>
__global__ void mul_kernel(const T* p1, const T* p2, T* out, size_t size)
{
    CPPGRAD_CUDA_1D_LOOP(i, size)
    {
        out[i] = p1[i] * p2[i];
    }
}

template <typename T>
__global__ void pow_kernel(const T* p1, const T* p2, T* out, size_t size)
{
    //CPPGRAD_CUDA_1D_LOOP(i, size)
    //{
    //    out[i] = pow(p1[i], p2[i]);
    //}
}

template <typename T>
__global__ void dot_kernel(const T* p1, const T* p2, T* out, size_t size)
{
    *out = T(0);

    CPPGRAD_CUDA_1D_LOOP(i, size)
    {
        *out += p1[i] * p2[i];
    }
}

template <typename T>
__global__ void matmul_kernel(const T* p1, const T* p2, T* out, size_t size)
{
}

}

#endif