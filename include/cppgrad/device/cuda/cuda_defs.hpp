// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_CUDA_DEFS_HPP
#define CPPGRAD_CUDA_DEFS_HPP

#include <algorithm>

namespace cppgrad {

namespace impl {
    constexpr unsigned int CPPGRAD_CUDA_NUM_THREADS = 128;
    constexpr unsigned int CPPGRAD_CUDA_MAX_GRID_SIZE = 4096;

    constexpr unsigned int CPPGRAD_CUDA_NUM_THREADS_2D_X = 16;
    constexpr unsigned int CPPGRAD_CUDA_NUM_THREADS_2D_Y = 16;

    constexpr unsigned int CPPGRAD_CUDA_MAX_GRID_SIZE_2D_X = 128;
    constexpr unsigned int CPPGRAD_CUDA_MAX_GRID_SIZE_2D_Y = 128;

    inline constexpr unsigned int grid_size_for_N(const unsigned int N)
    {
        return std::max(
            std::min((N + CPPGRAD_CUDA_NUM_THREADS - 1u) / CPPGRAD_CUDA_NUM_THREADS,
                CPPGRAD_CUDA_MAX_GRID_SIZE),
            1u);
    }
}

#ifdef __CUDACC__

namespace impl {

    /**
     * @brief Caffe2 inherited stuff
     *
     */

    inline dim3 grid_size_for_N_2D(const unsigned int Nx, const unsigned int Ny)
    {
        dim3 grid;

        grid.x = std::max(
            std::min((Nx + CPPGRAD_CUDA_NUM_THREADS_2D_X - 1u) / CPPGRAD_CUDA_NUM_THREADS_2D_X,
                CPPGRAD_CUDA_MAX_GRID_SIZE_2D_X),
            1u);

        grid.y = std::max(
            std::min((Ny + CPPGRAD_CUDA_NUM_THREADS_2D_Y - 1u) / CPPGRAD_CUDA_NUM_THREADS_2D_Y,
                CPPGRAD_CUDA_MAX_GRID_SIZE_2D_Y),
            1u);

        return grid;
    }

    inline dim3 grid_size_for_N_2D_mm(const unsigned int Nx, const unsigned int Ny, const unsigned int block_sz)
    {
        dim3 grid;

        grid.x = std::max(
            std::min((Nx + block_sz - 1u) / block_sz,
                CPPGRAD_CUDA_MAX_GRID_SIZE_2D_X),
            1u);

        grid.y = std::max(
            std::min((Ny + block_sz - 1u) / block_sz,
                CPPGRAD_CUDA_MAX_GRID_SIZE_2D_Y),
            1u);

        return grid;
    }

}

#define CPPGRAD_CUDA_1D_LOOP(idx, n)                                          \
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (n); \
         idx += blockDim.x * gridDim.x)

#define CPPGRAD_CUDA_2D_LOOP(ix, iy, nx, ny)                                     \
    for (unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; ix < (nx);     \
         ix += blockDim.x * gridDim.x)                                           \
        for (unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y; iy < (ny); \
             iy += blockDim.y * gridDim.y)

/**
 * @brief Macro to launch kernel using calculated grid size & default num threads.
 *
 */
#define CPPGRAD_CUDA_LAUNCH(kernel, count) \
    kernel<<<impl::grid_size_for_N(count), CPPGRAD_CUDA_NUM_THREADS>>>

#define CPPGRAD_CUDA_LAUNCH_2D(kernel, count_x, count_y) \
    kernel<<<impl::grid_size_for_N_2D(count_x, count_y), dim3(CPPGRAD_CUDA_NUM_THREADS_2D_X, CPPGRAD_CUDA_NUM_THREADS_2D_Y)>>>

/**
 * @brief This macro could be used to allow heterogeneous structs.
 * ATM used in StridedSpan.
 *
 */
#define CPPGRAD_CUDA_FN __host__ __device__
#else
#define CPPGRAD_CUDA_FN
#endif

}

#endif