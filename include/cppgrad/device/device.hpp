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
     * @param alignment alignment in bytes
     * @return std::byte* pointer to allocated chunk
     */
    virtual std::byte* allocate(std::size_t count, std::align_val_t alignment) = 0;

    /**
     * @brief Frees allocated memory chunk.
     * For CPU allocators it should pay attention to alignment.
     *
     * @param ptr chunk pointer
     * @param alignment alignment
     */
    virtual void deallocate(std::byte* ptr, std::align_val_t alignment) = 0;

    /**
     * @brief Get the Device executor object
     *
     * @return impl::Executor&
     */
    virtual impl::Executor& get_executor() = 0;

    /**
     * @brief Tells which type this Device is. "cpu", "cuda", etc..
     *
     * @return std::string_view type string
     */
    virtual std::string_view type() = 0;

    /**
     * @brief Clones device
     *
     * @return Device*
     */
    virtual Device* clone() const = 0;

    virtual ~Device() = default;
};

}

#endif