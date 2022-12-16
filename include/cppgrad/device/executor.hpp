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

enum CompareType {
    GT, // greater than
    LT, // less than
    GE, // greater than or equal
    LE, // less than or equal
    EQ, // equal
    NE, // not equal
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
    virtual void copy(const std::byte* from, std::byte* to,
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

    /**
     * @brief Assigns value of given DType to each Tensor element
     *
     * @param tensor tensor to fill
     * @param value pointer to value
     */
    virtual void fill(Tensor& tensor, std::byte* value) = 0;

    virtual void sum(const Tensor& lhs, const Tensor& rhs, Tensor& dst) = 0;
    virtual void sub(const Tensor& lhs, const Tensor& rhs, Tensor& dst) = 0;
    virtual void mul(const Tensor& lhs, const Tensor& rhs, Tensor& dst) = 0;
    virtual void matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst) = 0;
    virtual void relu(const Tensor& lhs, Tensor& dst) = 0;
    virtual void tanh(const Tensor& lhs, Tensor& dst) = 0;
    virtual void cmp(const Tensor& lhs, const Tensor& rhs, Tensor& dst, CompareType cmp_type) = 0;
};

}

#endif