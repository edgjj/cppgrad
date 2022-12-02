#ifndef CPPGRAD_CUDA_FILL_KERNEL_CUH
#define CPPGRAD_CUDA_FILL_KERNEL_CUH

namespace cppgrad {

namespace impl {

    template <typename T>
    __global__ void fill_kernel(T* data, size_t size, T val)
    {
        size_t linearIdx = blockIdx.x * blockDim.x + threadIdx.x;

        for (size_t i = linearIdx; i < size; i += gridDim.x * blockDim.x) {
            data[i] = val;
        }
    }

    template <typename T>
    static void fill_impl(std::byte* pos, std::byte* value, std::size_t count)
    {
        auto* ptr = reinterpret_cast<T*>(pos);
        auto fill_value = *reinterpret_cast<T*>(value);
        // fill_kernel<<<GRID_SZ, BLOCK_SZ>>>(ptr, count, fill_value);
    }

}

}

#endif