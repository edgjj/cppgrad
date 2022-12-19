#ifndef CPPGRAD_CUDA_DEFS_HPP
#define CPPGRAD_CUDA_DEFS_HPP

#include <algorithm>

namespace cppgrad {

namespace impl {

    /**
     * @brief Caffe2 inherited stuff
     *
     */

    constexpr unsigned int CPPGRAD_CUDA_NUM_THREADS = 128;
    constexpr unsigned int CPPGRAD_CUDA_MAX_GRID_SIZE = 4096;

    inline constexpr unsigned int grid_size_for_N(const unsigned int N)
    {
        return std::max(
            std::min((N + CPPGRAD_CUDA_NUM_THREADS - 1u) / CPPGRAD_CUDA_NUM_THREADS,
                CPPGRAD_CUDA_MAX_GRID_SIZE),
            1u);
    }

}

#define CPPGRAD_CUDA_1D_LOOP(idx, n)                                          \
    for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (n); \
         idx += blockDim.x * gridDim.x)

/**
 * @brief Macro to launch kernel using calculated grid size & default num threads.
 *
 */
#define CPPGRAD_CUDA_LAUNCH(kernel, count) \
    kernel<<<impl::grid_size_for_N(count), CPPGRAD_CUDA_NUM_THREADS>>>

/**
 * @brief This macro could be used to allow heterogeneous structs.
 * ATM used in StridedSpan.
 *
 */
#ifdef __CUDACC__
#define CPPGRAD_CUDA_FN __host__ __device__
#else
#define CPPGRAD_CUDA_FN
#endif

}

#endif