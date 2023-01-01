// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_DEVICE_CUDA_HPP
#define CPPGRAD_DEVICE_CUDA_HPP

#ifdef CPPGRAD_HAS_CUDA
#include "cppgrad/device/device.hpp"

namespace cppgrad {

struct CUDA : Device {

    std::byte* allocate(std::size_t count) override;
    void deallocate(std::byte* ptr) override;

    impl::Executor& get_executor() override;

    static int num_devices();
    static bool clear_cache();
    static void sync();
};

}

#endif

#endif