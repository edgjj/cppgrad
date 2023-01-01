// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "deps/bench_util.hpp"
#include <cppgrad/cppgrad.hpp>

#include <chrono>
#include <iostream>
#include <numeric>
#include <utility>

#define LOWER_BND 1
#define UPPER_BND 3

#define N_RUNS 10

using namespace cppgrad;

/*
    Dot product benchmark;
    Not accurate at all, since allocations are included in result;

    This is apparently a copy of bench_matmul;

    Changes:
        - 1-dim shape { vec_size } instead of 2-dim { mat_size, mat_size }
        - bench loop text message changed to dot product
*/

template <size_t DeviceTag>
void warmup_device(size_t n_runs)
{
    using TestResult = std::pair<double, double>;

    for (size_t i = 0; i < n_runs; i++) {
        auto t1 = Tensor::rand({ 128 }, LOWER_BND, UPPER_BND, f32, DeviceTag);

        t1 = t1.cpu();
    }
}

template <size_t DeviceTag>
void sync()
{
#ifdef CPPGRAD_HAS_CUDA
    if constexpr (DeviceTag == kCUDA) {
        CUDA::sync();
    }
#endif
}

using TestResult = std::pair<double, double>;

template <DType T, size_t DeviceTag>
TestResult bench_device(size_t vec_size, size_t n_runs)
{
    auto t1 = Tensor::rand({ vec_size }, LOWER_BND, UPPER_BND, T, DeviceTag);
    auto t2 = Tensor::rand({ vec_size }, LOWER_BND, UPPER_BND, T, DeviceTag);

    std::vector<double> run_times;

    for (size_t k = 0; k < n_runs; k++) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor t3 = cppgrad::mm(t1, t2);
        sync<DeviceTag>(); // force sync if cuda

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> run_ms = end - start;
        run_times.push_back(run_ms.count());
    }

    auto avg_time = std::reduce(run_times.begin(), run_times.end()) / run_times.size();
    // measure in GiB; replace 1 << 20 to 1e6 for GBs;
    auto avg_bandwidth = vec_size * sizeof(dtype_t<T>) / (avg_time * double(1 << 20));

    return { avg_time, avg_bandwidth };
}

int main()
{
    std::vector sizes { 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
    std::cout << "[ bench ] N runs per bench: " << N_RUNS << std::endl;
    std::cout << "[ bench ] Warming up... " << std::endl;

#ifdef CPPGRAD_HAS_CUDA
    warmup_device<kCUDA>(N_RUNS);
#endif

    for (auto& i : sizes) {
        std::cout << "[ bench ] Benchmarking " << i * i << " vector's dot product." << std::endl;

        auto [cpu_time, cpu_bw] = bench_device<f32, kCPU>(i * i, N_RUNS);
        std::cout << "\t[ CPU! ] avg_time: " << cpu_time << " ms; avg_bandwidth: " << cpu_bw << " GiB/s " << std::endl;

#ifdef CPPGRAD_HAS_CUDA
        auto [gpu_time, gpu_bw] = bench_device<f32, kCUDA>(i * i, N_RUNS);
        std::cout << "\t[ GPU (CUDA)! ] avg_time: " << gpu_time << " ms; avg_bandwidth: " << gpu_bw << " GiB/s " << std::endl;
        auto [speedup, efficiency] = bench::calc_metric_1d(gpu_time, cpu_time, i * i);
        std::cout << "\t[ Metrics ] Speedup: " << speedup << "x; Efficiency: " << efficiency << std::endl;
#endif
    }
}