#ifndef CPPGRAD_DEVICE_CPU_HPP
#define CPPGRAD_DEVICE_CPU_HPP

#include "cppgrad/device/device.hpp"

namespace cppgrad {

struct CPU : Device {

    std::byte* allocate(std::size_t count) override;
    void deallocate(std::byte* ptr) override;

    impl::Executor& get_executor() override;
};

}

#endif