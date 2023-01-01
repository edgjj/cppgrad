// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "cppgrad/device/cpu/cpu.hpp"
#include "cppgrad/device/cpu/cpu_executor.hpp"
#include "cppgrad/exceptions/out_of_memory.hpp"

#include <memory>

namespace cppgrad {

std::byte* CPU::allocate(std::size_t count)
{
    try {
        void* ptr = operator new[](count);
        return static_cast<std::byte*>(ptr);
    } catch (std::bad_alloc&) {
        throw exceptions::OutOfMemoryError("cpu", count);
    }
}

void CPU::deallocate(std::byte* ptr)
{
    operator delete[](ptr);
}

impl::Executor& CPU::get_executor()
{
    // dispatch between AVX/SSE/etc executors there?
    static impl::CPUExecutor executor;

    return executor;
}

}