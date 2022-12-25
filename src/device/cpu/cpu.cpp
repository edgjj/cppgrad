#include "cppgrad/device/cpu/cpu.hpp"
#include "cppgrad/device/cpu/cpu_executor.hpp"
#include "cppgrad/exceptions/out_of_memory.hpp"

#include <memory>

namespace cppgrad {

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

impl::Executor& CPU::get_executor()
{
    // dispatch between AVX/SSE/etc executors there?
    static impl::CPUExecutor executor;

    return executor;
}

Device* CPU::clone() const
{
    return new CPU();
}

std::string_view CPU::type()
{
    return "cpu";
}

}