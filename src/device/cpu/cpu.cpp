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
    static void strided_copy_impl(std::byte* from, std::byte* to, const std::vector<size_t>& shape, const std::vector<size_t>& strides)
    {
        auto* from_ptr = reinterpret_cast<T*>(from);
        auto* to_ptr = reinterpret_cast<T*>(to);

        // do something
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

void CPU::strided_copy(std::byte* from, std::byte* to, DType type, const std::vector<size_t>& shape, const std::vector<size_t>& strides)
{
    FOREACH_TYPE(type, impl::strided_copy_impl, from, to, shape, strides);
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