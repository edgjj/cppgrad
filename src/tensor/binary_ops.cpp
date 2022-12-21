#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/exceptions/type_error.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad {

namespace {
    void check_op_generic(const Tensor& lhs, const Tensor& rhs)
    {
        CPPGRAD_CHECK_EQ(lhs.dtype(), rhs.dtype(),
            exceptions::TypeError,
            lhs.dtype(),
            rhs.dtype());

        CPPGRAD_CHECK_EQ(lhs.dtype(), rhs.dtype(),
            exceptions::TypeError,
            lhs.dtype(),
            rhs.dtype());

        CPPGRAD_CHECK_EQ(lhs.device().type(), rhs.device().type(),
            exceptions::GenericError,
            "Device type mismatch");

        CPPGRAD_CHECK_EQ(lhs.get_align(), rhs.get_align(),
            exceptions::GenericError,
            "Tensors alignment mismatch");
    }

    void check_op_elementwise(const Tensor& lhs, const Tensor& rhs)
    {
        CPPGRAD_CHECK_EQ(lhs.numel(), rhs.numel(),
            exceptions::GenericError,
            "Shape mismatch");
    }
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs)
{
    auto out = lhs.clone();
    out += rhs;

    return out;
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs)
{
    auto out = lhs.clone();
    out -= rhs;

    return out;
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs)
{
    auto out = lhs.clone();
    out *= rhs;

    return out;
}

Tensor pow(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    auto out = Tensor::create_dirty(lhs.shape(), lhs.dtype(), lhs.get_align(), lhs.device().clone());
    auto& executor = out.device().get_executor();

    executor.pow(lhs, rhs, out);

    return out;
}

Tensor mm(const Tensor& lhs, const Tensor& rhs)
{
    CPPGRAD_CHECK_FALSE(lhs.shape().size() > 2 && rhs.shape().size() > 2,
        exceptions::GenericError,
        "MM supports only 1 & 2 dim Tensors atm");

    check_op_generic(lhs, rhs);

    // if not matmul
    if (lhs.shape() != rhs.shape() || lhs.shape().size() != 1) {
        CPPGRAD_CHECK_EQ(lhs.shape()[1], rhs.shape()[0],
            exceptions::GenericError,
            "LHS cols size not eq to RHS rows");

        auto out = Tensor::create_dirty({ lhs.shape()[0], rhs.shape()[1] }, lhs.dtype(), lhs.get_align(), lhs.device().clone());
        auto& executor = out.device().get_executor();
        executor.matmul(lhs, rhs, out);

        return out;
    } else {
        check_op_elementwise(lhs, rhs);
        auto out = Tensor::create_dirty({ 1 }, lhs.dtype(), lhs.get_align(), lhs.device().clone());
        auto& executor = out.device().get_executor();

        executor.dot(lhs, rhs, out);

        return out;
    }
}

}