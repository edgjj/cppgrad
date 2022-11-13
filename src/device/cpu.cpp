#include "cppgrad/device/cpu.hpp"

#include <memory>

namespace cppgrad {

void* CPU::allocate()
{
    return nullptr;
}

void CPU::deallocate(void* ptr)
{
}

void CPU::copy(void* from, void* to)
{
}

DeviceType CPU::type()
{
    return DeviceType::CPU;
}

static CPU cpu_device;
REGISTER_DEVICE("cpu", &cpu_device);

}