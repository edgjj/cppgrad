// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_CUDA_OPS_KERNELS_CUH
#define CPPGRAD_CUDA_OPS_KERNELS_CUH

#include "cppgrad/device/cuda/cuda_defs.hpp"
#include "cppgrad/tensor/util/strided_span.hpp"
#include <cuda.h>

#include <thrust/random.h>

namespace cppgrad::impl {

struct DotReduceTag { };
struct SumReduceTag { };

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
__global__ void random_fill_kernel(StridedSpan<T> out, double upper_bound, double lower_bound)
{
    thrust::default_random_engine engine(blockDim.x * blockIdx.x + threadIdx.x);

    CPPGRAD_CUDA_1D_LOOP(i, out.size())
    {
        engine.discard(i);
        if constexpr (std::is_integral_v<T>) {
            thrust::uniform_int_distribution<T> dist((T)lower_bound, (T)upper_bound);
            out[i] = dist(engine);
        } else {
            thrust::uniform_real_distribution<T> dist(lower_bound, upper_bound);
            out[i] = dist(engine);
        }
    }
}

template <typename T>
__global__ void add_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out)
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

template <typename T, typename Tag>
__global__ void reduce_kernel(ConstStridedSpan<T> p1, ConstStridedSpan<T> p2, StridedSpan<T> out, Tag tag)
{
    // we assume this kernel is always launched with CPPGRAD_CUDA_NUM_THREADS on block sz
    // 1 kb block shmem
    __shared__ T block_reduce[CPPGRAD_CUDA_NUM_THREADS];
    T* block_results = reinterpret_cast<T*>(staticmem::block_reduce);

    unsigned int local_idx = threadIdx.x;

    // zero-init; sdata
    block_reduce[local_idx] = T(0);

    // local reduction step
    CPPGRAD_CUDA_1D_LOOP(i, p1.size())
    {
        // ugly but too lazy to make lambdas work
        if constexpr (std::is_same_v<Tag, DotReduceTag>) {
            block_reduce[local_idx] += p1[i] * p2[i];
        } else if constexpr (std::is_same_v<Tag, SumReduceTag>) {
            block_reduce[local_idx] += p1[i];
        }
    }

    __syncthreads();

    // reduction 1/2
    for (unsigned int i = blockDim.x / 2; i != 0; i >>= 1) { // kernel 3
        if (local_idx < i) {
            block_reduce[local_idx] += block_reduce[local_idx + i];
        }

        __syncthreads();
    }

    if (local_idx == 0) {
        block_results[blockIdx.x] = block_reduce[0];
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
        out[0] = block_reduce[0];
    }
}

// tiled mm kernel
template <typename T, int BlkSize>
__global__ void matmul_kernel(ConstStridedSpan2D<T> p1, ConstStridedSpan2D<T> p2, StridedSpan2D<T> out)
{
    // CPPGRAD_CUDA_NUM_THREADS_2D_X/Y is tile sz
    __shared__ T tile_p1[BlkSize][BlkSize];
    __shared__ T tile_p2[BlkSize][BlkSize];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int n_tiles = 1 + ((p2.size(0) - 1) / BlkSize); // or (p1.size(1))

    auto tX = threadIdx.x;
    auto tY = threadIdx.y;

    T temp { 0 };

    auto h1 = p1.size(0),
         h2 = p2.size(0),
         w1 = p1.size(1),
         w2 = p2.size(1);

    tile_p1[tY][tX] = 0;
    tile_p2[tY][tX] = 0;

    __syncthreads();

    for (unsigned int t = 0; t < n_tiles; t++) {

        auto tileStep = t * BlkSize;
        auto tStepX = tileStep + tX,
             tStepY = tileStep + tY;

        if (row < h1 && tStepX < w1) {
            tile_p1[tY][tX] = p1(row, tStepX);
        }

        if (col < w2 && tStepY < h2) {
            tile_p2[tY][tX] = p2(tStepY, col);
        }

        __syncthreads();

#pragma unroll
        for (unsigned int k = 0; k < BlkSize; k++) {
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