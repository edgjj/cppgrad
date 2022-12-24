#ifndef CPPGRAD_CUDA_OPS_KERNELS_CUH
#define CPPGRAD_CUDA_OPS_KERNELS_CUH

#include "cppgrad/device/cuda/cuda_defs.hpp"
#include "cppgrad/tensor/util/strided_span.hpp"
#include <cuda.h>

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
__global__ void div_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        out[i] = p1[i] / p2[i];
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
    // 1 kb block shmem
    __shared__ T block_dot[CPPGRAD_CUDA_NUM_THREADS];
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

// tiled mm kernel
template <typename T>
__global__ void matmul_kernel(ConstStridedSpan2D<T> p1, ConstStridedSpan2D<T> p2, StridedSpan2D<T> out)
{
    // CPPGRAD_CUDA_NUM_THREADS_2D_X/Y is tile sz
    __shared__ T tile_p1[CPPGRAD_CUDA_NUM_THREADS_2D_X][CPPGRAD_CUDA_NUM_THREADS_2D_Y];
    __shared__ T tile_p2[CPPGRAD_CUDA_NUM_THREADS_2D_X][CPPGRAD_CUDA_NUM_THREADS_2D_Y];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int n_tiles = 1 + ((p2.size(0) - 1) / CPPGRAD_CUDA_NUM_THREADS_2D_X); // or (p1.size(1))

    auto tX = threadIdx.x;
    auto tY = threadIdx.y;

    T temp { 0 };
    for (unsigned int t = 0; t < n_tiles; t++) {

        tile_p1[tY][tX] = row < p1.size(0) && t * CPPGRAD_CUDA_NUM_THREADS_2D_X + tX < p1.size(1)
            ? p1(row, t * CPPGRAD_CUDA_NUM_THREADS_2D_X + tX)
            : T(0);

        tile_p2[tY][tX] = t * CPPGRAD_CUDA_NUM_THREADS_2D_Y + tY < p2.size(0) && col < p2.size(1)
            ? p2(t * CPPGRAD_CUDA_NUM_THREADS_2D_Y + tY, col)
            : T(0);

        __syncthreads();

        for (unsigned int k = 0; k < CPPGRAD_CUDA_NUM_THREADS_2D_X; k++) {
            temp += tile_p1[tY][k] * tile_p2[k][tX];
        }

        __syncthreads();
    }

    if (row < out.size(0) && col < out.size(1)) {
        out(row, col) = temp;
    }
}

// lhs to out kernels

template <typename T>
__global__ void log_kernel(ConstStridedSpan<T> p1, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        // escape to global namespace to avoid cppgrad::pow
        if constexpr (std::is_same_v<double, T>) {
            out[i] = ::log(p1[i]);
        } else {
            out[i] = ::logf(p1[i]);
        }
    }
}

template <typename T>
__global__ void exp_kernel(ConstStridedSpan<T> p1, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        // escape to global namespace to avoid cppgrad::pow
        if constexpr (std::is_same_v<double, T>) {
            out[i] = ::exp(p1[i]);
        } else {
            out[i] = ::expf(p1[i]);
        }
    }
}

template <typename T>
__global__ void relu_kernel(ConstStridedSpan<T> p1, StridedSpan<T> out) // can be synthesized from max/min
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        // escape to global namespace to avoid cppgrad::pow
        out[i] = ::max(T(0), (p1[i]));
    }
}

template <typename T>
__global__ void tanh_kernel(ConstStridedSpan<T> p1, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        if constexpr (std::is_same_v<double, T>) {
            out[i] = ::tanh(p1[i]);
        } else {
            out[i] = ::tanhf(p1[i]);
        }
    }
}

template <typename T>
__global__ void sign_kernel(ConstStridedSpan<T> p1, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        if constexpr (!std::is_signed_v<T>) {
            out[i] = T(0) < p1[i];
        } else {
            out[i] = (T(0) < p1[i]) - (p1[i] < T(0));
        }
    }
}

template <typename T>
__global__ void neg_kernel(ConstStridedSpan<T> p1, StridedSpan<T> out)
{
    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        if constexpr (!std::is_signed_v<T>) {
            out[i] = p1[i]; // no op on unsigned types
        } else {
            out[i] = -p1[i];
        }
    }
}

}

#endif