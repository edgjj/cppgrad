#ifndef CPPGRAD_DEVICE_CUDA_HPP
#define CPPGRAD_DEVICE_CUDA_HPP

#ifdef CPPGRAD_HAS_CUDA
#include "cppgrad/device/device.hpp"

namespace cppgrad {

struct CUDA : Device {

    std::byte* allocate(std::size_t count, std::align_val_t alignment) override;
    void deallocate(std::byte* ptr, std::align_val_t alignment) override;

    impl::Executor& get_executor() override;
    Device* clone() override;

    std::string_view type() override;

    static int num_devices();
};

}

#endif

#endif