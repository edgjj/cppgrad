#include "cppgrad/device/cpu.hpp"
#include "cppgrad/device/registry.hpp"

#include <cstring>
#include <memory>

namespace cppgrad {

std::byte* CPU::allocate(std::size_t count, std::align_val_t alignment)
{
    void* ptr = operator new[](count * sizeof(std::byte), alignment);
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

REGISTER_DEVICE(CPU, "cpu");

}