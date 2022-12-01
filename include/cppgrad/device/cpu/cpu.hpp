#ifndef CPPGRAD_DEVICE_CPU_HPP
#define CPPGRAD_DEVICE_CPU_HPP

#include "cppgrad/device/device.hpp"

namespace cppgrad {

struct CPU : public Device {

    [[nodiscard]] std::byte* allocate(std::size_t count, std::align_val_t alignment, std::string& err) override;
    void deallocate(std::byte* ptr, std::align_val_t alignment) override;

    void copy(std::byte* from, std::byte* to, std::size_t count) override;
    void assign(std::byte* pos, std::byte* value, DType type) override;
    void fill(std::byte* pos, std::byte* value, DType type, std::size_t count) override;

    std::string_view type() override;
};

}

#endif