#ifndef CPPGRAD_DEVICE_EXECUTOR_HPP
#define CPPGRAD_DEVICE_EXECUTOR_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"
#include "cppgrad/tensor/typing.hpp"

#include <cstddef>
#include <vector>

namespace cppgrad::impl {

enum CopyType {
    Homogeneous = 1,
    HostToDevice = 2,
    DeviceToHost = 4
};

struct Executor {
    /**
     * @brief Non-strided copy routine. Shouldn't be used with non-contiguous chunks.
     * Root copy function, should allow for h2d, d2h, and homogeneous copies.
     *
     * @param from source chunk
     * @param to destination chunk
     * @param count size in bytes
     * @param copy_type copy type, see enum CopyType
     */
    virtual void copy(std::byte* from, std::byte* to,
        std::size_t count,
        CopyType copy_type = impl::Homogeneous)
        = 0;

    /**
     * @brief Strided row-major copy routine
     *
     * @param from source Tensor
     * @param to destination Tensor
     */
    virtual void strided_copy(const Tensor& from, Tensor& to) = 0;

    // TODO: think about using this for assigning scalars/vectors without using intermediate Tensors.
    // virtual void assign(std::byte* pos, std::byte* value, DType type, std::size_t count) = 0;

    /**
     * @brief Assigns value of given DType to each Tensor element
     *
     * @param tensor tensor to fill
     * @param value pointer to value
     */
    virtual void fill(Tensor& tensor, std::byte* value) = 0;
};

}

#endif