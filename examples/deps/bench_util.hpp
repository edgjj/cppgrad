// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_EXAMPLES_BENCH_UTIL_HPP
#define CPPGRAD_EXAMPLES_BENCH_UTIL_HPP

#include "cppgrad/device/cuda/cuda_defs.hpp"
#include <algorithm>
#include <cstddef>
#include <thread>
#include <utility>

namespace bench {

std::pair<double, double> calc_metric_1d(double time_cuda, double time_cpu, size_t data_N)
{
    using namespace cppgrad;

    auto n_blocks = cppgrad::impl::grid_size_for_N(data_N);

    auto total_cuda_proc = n_blocks * impl::CPPGRAD_CUDA_NUM_THREADS;
    auto real_used_cuda_proc = n_blocks == impl::CPPGRAD_CUDA_MAX_GRID_SIZE ? total_cuda_proc : data_N;

    auto n_t = (double)real_used_cuda_proc / (double)std::thread::hardware_concurrency();
    auto speedup = time_cpu / time_cuda;

    // return speedup & efficiency; this might not be actual efficiency
    return std::make_pair(speedup, speedup / n_t);
}

std::pair<double, double> calc_metric_2d(double time_cuda, double time_cpu, size_t data_Nx, size_t data_Ny)
{
    using namespace cppgrad;

    auto n_blocks_x = std::max(
        std::min(((unsigned int)data_Nx + impl::CPPGRAD_CUDA_NUM_THREADS_2D_X - 1u) / impl::CPPGRAD_CUDA_NUM_THREADS_2D_X,
            impl::CPPGRAD_CUDA_MAX_GRID_SIZE_2D_X),
        1u);

    auto n_blocks_y = std::max(
        std::min(((unsigned int)data_Ny + impl::CPPGRAD_CUDA_NUM_THREADS_2D_X - 1u) / impl::CPPGRAD_CUDA_NUM_THREADS_2D_X,
            impl::CPPGRAD_CUDA_MAX_GRID_SIZE_2D_X),
        1u);

    auto total_cuda_proc = n_blocks_x * n_blocks_y * impl::CPPGRAD_CUDA_NUM_THREADS_2D_X * impl::CPPGRAD_CUDA_NUM_THREADS_2D_Y;
    auto real_used_cuda_proc = n_blocks_x * n_blocks_y == impl::CPPGRAD_CUDA_MAX_GRID_SIZE_2D_X * impl::CPPGRAD_CUDA_MAX_GRID_SIZE_2D_Y ? total_cuda_proc : data_Nx * data_Ny;

    auto n_t = (double)real_used_cuda_proc / (double)std::thread::hardware_concurrency();
    auto speedup = time_cpu / time_cuda;

    // return speedup & efficiency
    return std::make_pair(speedup, speedup / n_t);
}

}

#endif