#include "cppgrad/device/cpu/cpu.hpp"

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

}