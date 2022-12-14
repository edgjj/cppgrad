#ifndef CPPGRAD_DEVICE_CPU_HPP
#define CPPGRAD_DEVICE_CPU_HPP

#include "cppgrad/device/device.hpp"

namespace cppgrad {

struct CPU : public Device {

    std::byte* allocate(std::size_t count, std::align_val_t alignment) override;
    void deallocate(std::byte* ptr, std::align_val_t alignment) override;

    void copy(std::byte* from, std::byte* to, std::size_t count) override;
    void strided_copy(std::byte* from,
        std::byte* to,
        DType type,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& from_strides,
        const std::vector<size_t>& to_strides) override;

    // void assign(std::byte* pos, std::byte* value, DType type, std::size_t count) override;
    void fill(std::byte* pos, std::byte* value, DType type, std::size_t count) override;

    std::string_view type() override;
};

}

#endif