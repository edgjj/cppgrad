#include <chrono>
#include <cppgrad/cppgrad.hpp>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>

#define LOWER_BND 1
#define UPPER_BND 10

#define N_RUNS 10

using namespace cppgrad;

/*
    Matmul benchmark;
    Not accurate at all, since allocations are included in result;
*/

template <typename T>
std::vector<T> get_random_vec(size_t num_elements, T upper_bound, T lower_bound)
{
    std::mt19937 engine(std::random_device {}());

    std::vector<T> data;
    data.reserve(num_elements);

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist { lower_bound, upper_bound };

        for (size_t i = 0; i < num_elements; i++) {
            data.push_back(dist(engine));
        }

        return std::move(data);
    } else {
        std::uniform_int_distribution<T> dist { lower_bound, upper_bound };

        for (size_t i = 0; i < num_elements; i++) {
            data.push_back(dist(engine));
        }

        return std::move(data);
    }
}

template <typename DeviceType>
void warmup_device(size_t n_runs)
{
    using TestResult = std::pair<double, double>;

    for (size_t i = 0; i < n_runs; i++) {
        auto v = get_random_vec<dtype_t<i32>>(128, LOWER_BND, UPPER_BND);
        auto t1 = Tensor::from_blob<i32, DeviceType>(v.data(), { 128 });

        t1 = t1.cpu();
    }
}

template <typename DeviceType>
void sync()
{
#ifdef CPPGRAD_HAS_CUDA
    if constexpr (std::is_same_v<DeviceType, CUDA>) {
        CUDA::sync();
    }
#endif
}

using TestResult = std::pair<double, double>;

template <DType T, typename DeviceType>
TestResult bench_device(size_t mat_size, size_t n_runs)
{
    auto v1 = get_random_vec<dtype_t<T>>(mat_size * mat_size, LOWER_BND, UPPER_BND);
    auto v2 = get_random_vec<dtype_t<T>>(mat_size * mat_size, LOWER_BND, UPPER_BND);

    auto t1 = Tensor::from_blob<T, DeviceType>(v1.data(), { mat_size, mat_size });
    auto t2 = Tensor::from_blob<T, DeviceType>(v2.data(), { mat_size, mat_size });

    std::vector<double> run_times;

    for (size_t k = 0; k < n_runs; k++) {
        auto start = std::chrono::high_resolution_clock::now();
        Tensor t3 = cppgrad::mm(t1, t2);
        sync<DeviceType>(); // force sync
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> run_ms = end - start;
        run_times.push_back(run_ms.count());
    }

    auto avg_time = std::reduce(run_times.begin(), run_times.end()) / run_times.size();
    // measure in GiB; replace 1 << 20 to 1e6 for GBs;
    auto avg_bandwidth = mat_size * mat_size * sizeof(dtype_t<T>) / (avg_time * double(1 << 20));

    return { avg_time, avg_bandwidth };
}

int main()
{
    std::vector sizes { 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };
    std::cout << "[ bench ] N runs per bench: " << N_RUNS << std::endl;
    std::cout << "[ bench ] Warming up... " << std::endl;

#ifdef CPPGRAD_HAS_CUDA
    warmup_device<CUDA>(N_RUNS);
#endif

    for (auto& i : sizes) {
        std::cout << "[ bench ] Benchmarking " << i << "x" << i << " matrix multiply." << std::endl;

        auto [cpu_time, cpu_bw] = bench_device<f32, CPU>(i, N_RUNS);
        std::cout << "\t[ bench ] CPU! avg_time: " << cpu_time << " ms; avg_bandwidth: " << cpu_bw << " GiB/s " << std::endl;

#ifdef CPPGRAD_HAS_CUDA
        auto [gpu_time, gpu_bw] = bench_device<f32, CUDA>(i, N_RUNS);
        std::cout << "\t[ bench ] GPU (CUDA)! avg_time: " << gpu_time << " ms; avg_bandwidth: " << gpu_bw << " GiB/s " << std::endl;
#endif
    }
}