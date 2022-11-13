#ifndef CPPGRAD_DEVICE_CPU_HPP
#define CPPGRAD_DEVICE_CPU_HPP

#include "cppgrad/device/device.hpp"

namespace cppgrad {

struct CPU : public Device {

    void* allocate() override;
    void deallocate(void* ptr) override;
    void copy(void* from, void* to) override;

    DeviceType type() override;
};

}

#endif