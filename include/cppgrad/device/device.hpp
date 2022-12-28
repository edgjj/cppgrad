#ifndef CPPGRAD_DEVICE_HPP
#define CPPGRAD_DEVICE_HPP

#include "cppgrad/device/executor.hpp"
#include "cppgrad/tensor/tensor_fwd.hpp"

#include <cstddef>
#include <new>
#include <string_view>

namespace cppgrad {

struct Device {

    /**
     * @brief Retrieves memory chunk of specified size & alignment.
     *
     * @param count size in bytes
     * @return std::byte* pointer to allocated chunk
     */
    virtual std::byte* allocate(std::size_t count) = 0;

    /**
     * @brief Frees allocated memory chunk.
     * For CPU allocators it should pay attention to alignment.
     *
     * @param ptr chunk pointer
     * @param alignment alignment
     */
    virtual void deallocate(std::byte* ptr) = 0;

    /**
     * @brief Get the Device executor object
     *
     * @return impl::Executor&
     */
    virtual impl::Executor& get_executor() = 0;

    virtual ~Device() = default;
};

}

#endif