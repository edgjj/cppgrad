#ifndef CPPGRAD_CUDA_STRIDED_COPY_KERNEL_CUH
#define CPPGRAD_CUDA_STRIDED_COPY_KERNEL_CUH

namespace cppgrad {

namespace impl {

    template <typename T>
    __global__ void strided_copy_kernel(T* from, T* to, size_t* shape, size_t* strides, size_t shape_size)
    {
        size_t linearIdx = blockIdx.x * blockDim.x + threadIdx.x;

        // for (size_t i = linearIdx; i < size; i += gridDim.x * blockDim.x) {
        //     data[i] = val;
        // }
    }

    template <typename T>
    static void strided_copy_impl(std::byte* from,
        std::byte* to,
        const size_t* shape,
        const size_t* from_strides,
        const size_t* to_strides, size_t shape_size)
    {
        throw std::exception();

        // auto* ptr = reinterpret_cast<T*>(pos);
        // auto fill_value = *reinterpret_cast<T*>(value);
        // strided_copy_kernel<<<GRID_SZ, BLOCK_SZ>>>(ptr, count, fill_value);
    }

}

}

#endif