#ifndef CPPGRAD_CUDA_FILL_KERNEL_CUH
#define CPPGRAD_CUDA_FILL_KERNEL_CUH

#include "cppgrad/device/cuda/cuda_defs.hpp"

namespace cppgrad {

namespace impl {

    template <typename T>
    __global__ void fill_kernel(T* data, size_t size, T val)
    {
        CPPGRAD_CUDA_1D_LOOP(i, size)
        {
            data[i] = val;
        }
    }

    template <typename T>
    static void fill_impl(std::byte* pos, std::byte* value, std::size_t count)
    {
        auto* ptr = reinterpret_cast<T*>(pos);
        auto fill_value = *reinterpret_cast<T*>(value);

        CPPGRAD_CUDA_LAUNCH(fill_kernel, count)
        (ptr, count, fill_value);
    }

}

}

#endif