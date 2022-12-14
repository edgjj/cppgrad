#ifndef CPPGRAD_CUDA_DEFS_HPP
#define CPPGRAD_CUDA_DEFS_HPP

#include <algorithm>

namespace cppgrad {

namespace impl {

    /**
     * @brief Caffe2 inherited stuff
     *
     */

    constexpr int CPPGRAD_CUDA_NUM_THREADS = 128;
    constexpr int CPPGRAD_CUDA_MAX_GRID_SIZE = 4096;

    inline constexpr int grid_size_for_N(const int N)
    {
        return std::max(
            std::min((N + CPPGRAD_CUDA_NUM_THREADS - 1) / CPPGRAD_CUDA_NUM_THREADS,
                CPPGRAD_CUDA_MAX_GRID_SIZE),
            1);
    }

}

#define CPPGRAD_CUDA_1D_LOOP(idx, n)                                    \
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (n); \
         idx += blockDim.x * gridDim.x)

/**
 * @brief Macro to launch kernel using calculated grid size & default num threads.
 *
 */
#define CPPGRAD_CUDA_LAUNCH(kernel, count) \
    kernel<<<impl::grid_size_for_N(count), CPPGRAD_CUDA_NUM_THREADS>>>

}

#endif