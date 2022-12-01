#ifndef CPPGRAD_DEVICE_HPP
#define CPPGRAD_DEVICE_HPP

#include "cppgrad/tensor/typing.hpp"

#include <cstddef>
#include <new>
#include <string>
#include <string_view>

namespace cppgrad {

struct Device {

    virtual std::byte* allocate(std::size_t count, std::align_val_t alignment, std::string& err) = 0;
    virtual void deallocate(std::byte* ptr, std::align_val_t alignment) = 0;
    virtual void copy(std::byte* from, std::byte* to, std::size_t count) = 0;
    virtual void assign(std::byte* pos, std::byte* value, DType type) = 0;
    virtual void fill(std::byte* pos, std::byte* value, DType type, std::size_t count) = 0;

    virtual std::string_view type() = 0;

    virtual ~Device() = default;
};

}

#endif