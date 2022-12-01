#include "cppgrad/device/cpu/cpu.hpp"

#include <algorithm>
#include <cstring>
#include <memory>


namespace cppgrad {

std::byte* CPU::allocate(std::size_t count, std::align_val_t alignment)
{
    void* ptr = operator new[](count, alignment);
    return static_cast<std::byte*>(ptr);
}

void CPU::deallocate(std::byte* ptr, std::align_val_t alignment)
{
    operator delete[](ptr, alignment);
}

void CPU::copy(std::byte* from, std::byte* to, std::size_t count)
{
    std::memcpy(to, from, count);
}

void CPU::assign(std::byte* pos, std::byte* value, DType type)
{
    copy(value, pos, type_size(type));
}

template <typename T>
static void fill_internal(std::byte* pos, std::byte* value, std::size_t count)
{
    auto* ptr = reinterpret_cast<T*>(pos);
    auto fill_value = *reinterpret_cast<T*>(value);

    // std::fill_n(OutputIt, Size, T& value)
    std::fill_n(ptr, count, fill_value);
}

void CPU::fill(std::byte* pos, std::byte* value, DType type, std::size_t count)
{
    switch (type) {
    case u32:
        fill_internal<dtype_t<u32>>(pos, value, count);
        break;
    case u64:
        fill_internal<dtype_t<u32>>(pos, value, count);
        break;
    case i32:
        fill_internal<dtype_t<u32>>(pos, value, count);
        break;
    case i64:
        fill_internal<dtype_t<u32>>(pos, value, count);
        break;
    case f32:
        fill_internal<dtype_t<u32>>(pos, value, count);
        break;
    case f64:
        fill_internal<dtype_t<u32>>(pos, value, count);
        break;
    }
}

}