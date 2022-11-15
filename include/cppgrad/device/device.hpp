#ifndef CPPGRAD_DEVICE_HPP
#define CPPGRAD_DEVICE_HPP

#include <cstddef>
#include <new>

namespace cppgrad {

struct Device {

    virtual std::byte* allocate(std::size_t count, std::align_val_t alignment) = 0;
    virtual void deallocate(std::byte* ptr, std::align_val_t alignment) = 0;
    virtual void copy(std::byte* from, std::byte* to, std::size_t count) = 0;
};

}

#endif