#ifndef CPPGRAD_DEVICE_CUDA_HPP
#define CPPGRAD_DEVICE_CUDA_HPP

#ifdef CPPGRAD_HAS_CUDA
#include "cppgrad/device/device.hpp"

namespace cppgrad {

struct CUDA : public Device {
    std::byte* allocate(std::size_t count, std::align_val_t alignment) override;
    void deallocate(std::byte* ptr, std::align_val_t alignment) override;
    void copy(std::byte* from, std::byte* to, std::size_t count) override;

    void copy_from_host(std::byte* from, std::byte* to, std::size_t count);
    void copy_to_host(std::byte* from, std::byte* to, std::size_t count);

    static int num_devices();
};

}

#endif

#endif