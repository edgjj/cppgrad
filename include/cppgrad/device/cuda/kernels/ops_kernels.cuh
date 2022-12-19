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
__global__ void fill_kernel(StridedSpan<T> out, T val)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        out[i] = val;
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

namespace staticmem {
    // statically alloc 64k for inter-block reduce
    __device__ static std::byte block_reduce[impl::CPPGRAD_CUDA_MAX_GRID_SIZE * 16];
}

template <typename T>
__global__ void dot_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out)
{
    // we assume this kernel is always launched with CPPGRAD_CUDA_NUM_THREADS on block sz
    __shared__ T block_dot[impl::CPPGRAD_CUDA_NUM_THREADS];
    T* block_results = reinterpret_cast<T*>(staticmem::block_reduce);

    unsigned int local_idx = threadIdx.x;

    // zero-init; sdata
    block_dot[local_idx] = T(0);

    CPPGRAD_CUDA_1D_LOOP(i, p1.size())
    {
        block_dot[local_idx] += p1[i] * p2[i];
    }

    __syncthreads();

    // reduction 1/2
    for (unsigned int i = blockDim.x / 2; i != 0; i >>= 1) { // kernel 3
        if (local_idx < i) {
            block_dot[local_idx] += block_dot[local_idx + i];
        }

        __syncthreads();
    }

    if (local_idx == 0) {
        block_results[blockIdx.x] = block_dot[0];
    }

    if (gridDim.x > 1) {
        __threadfence();

        if (local_idx == 0) {
            unsigned int& retirementCount = *reinterpret_cast<unsigned int*>(out.data());
            auto ticket = atomicInc(&retirementCount, gridDim.x); // ((old >= val) ? 0 : (old+1))

            // check if block is actually last; reduce by last thread available
            if (ticket == gridDim.x - 1) {
                out[0] = 0;
                for (unsigned int i = 0; i < gridDim.x; i++) {
                    out[0] += block_results[i];
                }
            }
        }
    } else {
        // just copy value to global mem
        out[0] = block_dot[0];
    }
}

template <typename T>
__global__ void matmul_kernel(ConstStridedSpan2D<T> p1, ConstStridedSpan2D<T> p2, StridedSpan2D<T> out)
{
}

}

#endif