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

Tensor& operator+=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    auto& executor = lhs.device().get_executor();
    executor.sum(lhs, rhs, lhs);

    return lhs;
}

Tensor& operator-=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    auto& executor = lhs.device().get_executor();
    executor.sub(lhs, rhs, lhs);

    return lhs;
}

Tensor& operator*=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    auto& executor = lhs.device().get_executor();
    executor.mul(lhs, rhs, lhs);

    return lhs;
}

}