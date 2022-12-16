#ifndef CPPGRAD_CUDA_STRIDED_COPY_KERNEL_CUH
#define CPPGRAD_CUDA_STRIDED_COPY_KERNEL_CUH

#include "cppgrad/device/cuda/cuda_defs.hpp"

namespace cppgrad {

namespace impl {

    template <typename T>
    __global__ void strided_copy_kernel(const T* from,
        T* to,
        size_t count,
        size_t from_stride,
        size_t to_stride)
    {
        CPPGRAD_CUDA_1D_LOOP(i, count)
        {
            to[i * to_stride] = from[i * from_stride];
        }
    }

    template <typename T>
    static void strided_copy_impl(const std::byte* from,
        std::byte* to,
        const size_t* shape,
        const size_t* from_strides,
        const size_t* to_strides,
        size_t shape_size)
    {
        if (shape_size == 1) {
            CPPGRAD_CUDA_LAUNCH(strided_copy_kernel, *shape)
            (reinterpret_cast<const T*>(from),
                reinterpret_cast<T*>(to),
                *shape,
                *from_strides / sizeof(T),
                *to_strides / sizeof(T));

            return;
        }

        while (shape_size != 1) {
            strided_copy_impl<T>(from + *from_strides, to + *to_strides,
                ++shape,
                ++from_strides,
                ++to_strides,
                --shape_size);
        }
    }

}

}

#endif