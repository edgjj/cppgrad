#ifndef CPPGRAD_DEVICE_HPP
#define CPPGRAD_DEVICE_HPP

#include "cppgrad/tensor/typing.hpp"

#include <cstddef>
#include <new>
#include <string>
#include <string_view>
#include <vector>

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
     * @brief Non-strided copy routine. Shouldn't be used with non-contiguous chunks.
     *
     * @param from source chunk
     * @param to destination chunk
     * @param count size in bytes
     */
    virtual void copy(std::byte* from, std::byte* to, std::size_t count) = 0;

    /**
     * @brief Strided row-major copy routine
     *
     * @param from source data chunk
     * @param to destination data chunk
     * @param type data DType
     * @param shape data shape
     * @param strides data strides
     */
    virtual void strided_copy(std::byte* from, std::byte* to, DType type, const std::vector<size_t>& shape, const std::vector<size_t>& strides) = 0;

    // TODO: think about using this for assigning scalars/vectors without using intermediate Tensors.
    // virtual void assign(std::byte* pos, std::byte* value, DType type, std::size_t count) = 0;

    /**
     * @brief Assigns value of given DType to each chunk element
     *
     * @param pos pointer to chunk/position
     * @param value pointer to value
     * @param type data DType
     * @param count size in elements
     */
    virtual void fill(std::byte* pos, std::byte* value, DType type, std::size_t count) = 0;

    /**
     * @brief Tells which type this Device is. "cpu", "cuda", etc..
     *
     * @return std::string_view type string
     */
    virtual std::string_view type() = 0;

    virtual ~Device() = default;
};

}

#endif