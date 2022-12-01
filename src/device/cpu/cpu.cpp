#include "cppgrad/device/cpu/cpu.hpp"

#include <algorithm>
#include <cstring>
#include <memory>

namespace cppgrad {

std::byte* CPU::allocate(std::size_t count, std::align_val_t alignment, std::string& err)
{
    try {
        void* ptr = operator new[](count, alignment);
        return static_cast<std::byte*>(ptr);
    } catch (std::bad_alloc&) {
        // this is ugly
        err += "[ ";
        err += type();
        err += " ]";
        err += "Device out of memory. Tried to allocate: ";
        err += std::to_string(count);
        err += " bytes";
        return nullptr;
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
    FOREACH_TYPE(type, fill_internal, pos, value, count);
}

std::string_view CPU::type()
{
    return "cpu";
}

}