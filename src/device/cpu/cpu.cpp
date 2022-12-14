#include "cppgrad/device/cpu/cpu.hpp"
#include "cppgrad/exceptions/out_of_memory.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace cppgrad {

namespace impl {

    template <typename T>
    static void fill_impl(std::byte* pos, std::byte* value, std::size_t count)
    {
        auto* ptr = reinterpret_cast<T*>(pos);
        auto fill_value = *reinterpret_cast<T*>(value);

        // std::fill_n(OutputIt, Size, T& value)
        std::fill_n(ptr, count, fill_value);
    }

    template <typename T>
    static void strided_copy_copier(T* from,
        T* to,
        size_t count,
        size_t from_stride,
        size_t to_stride)
    {
        for (size_t i = 0; i < count; i++) {
            to[i * to_stride] = from[i * from_stride];
        }
    }

    template <typename T>
    static void strided_copy_impl(std::byte* from,
        std::byte* to,
        const size_t* shape,
        const size_t* from_strides,
        const size_t* to_strides,
        size_t shape_size)
    {
        if (shape_size == 1) {
            strided_copy_copier<T>(reinterpret_cast<T*>(from),
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

std::byte* CPU::allocate(std::size_t count, std::align_val_t alignment)
{
    try {
        void* ptr = operator new[](count, alignment);
        return static_cast<std::byte*>(ptr);
    } catch (std::bad_alloc&) {
        throw exceptions::OutOfMemoryError(type(), count);
    }
}

void CPU::deallocate(std::byte* ptr, std::align_val_t alignment)
{
    operator delete[](ptr, alignment);
}

void CPU::copy(std::byte* from, std::byte* to, std::size_t count)
{
    std::memcpy(to, from, count);
}

void CPU::strided_copy(std::byte* from,
    std::byte* to,
    DType type,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& from_strides,
    const std::vector<size_t>& to_strides)
{
    FOREACH_TYPE(type, impl::strided_copy_impl, from, to, shape.data(), from_strides.data(), to_strides.data(), shape.size());
}

// void CPU::assign(std::byte* pos, std::byte* value, DType type, std::size_t count)
// {
//     copy(value, pos, dtype_size(type) * count);
// }

void CPU::fill(std::byte* pos, std::byte* value, DType type, std::size_t count)
{
    FOREACH_TYPE(type, impl::fill_impl, pos, value, count);
}

std::string_view CPU::type()
{
    return "cpu";
}

}