#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/exceptions/type_error.hpp"
#include "cppgrad/tensor/ops/grad_ops.hpp"
#include "cppgrad/tensor/tensor.hpp" // tensor.hpp includes op_overloads.hpp

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

    auto lhs_out = lhs.requires_grad() ? lhs.clone() : lhs;
    lhs = AddOp::apply({ lhs_out, rhs })[0];

    return lhs;
}

Tensor& operator-=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    auto lhs_out = lhs.requires_grad() ? lhs.clone() : lhs;
    lhs = SubOp::apply({ lhs_out, rhs })[0];

    return lhs;
}

Tensor& operator*=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    auto lhs_out = lhs.requires_grad() ? lhs.clone() : lhs;
    lhs = MultiplyOp::apply({ lhs_out, rhs })[0];

    return lhs;
}

Tensor& operator/=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    auto lhs_out = lhs.requires_grad() ? lhs.clone() : lhs;
    lhs = DivisionOp::apply({ lhs_out, rhs })[0];

    return lhs;
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return AddOp::apply({ lhs.clone(), rhs })[0];
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return SubOp::apply({ lhs.clone(), rhs })[0];
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return MultiplyOp::apply({ lhs.clone(), rhs })[0];
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return DivisionOp::apply({ lhs.clone(), rhs })[0];
}

Tensor pow(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return PowOp::apply({ lhs.clone(), rhs })[0];
}

Tensor mm(const Tensor& lhs, const Tensor& rhs)
{
    CPPGRAD_CHECK_FALSE(lhs.shape().size() > 2 && rhs.shape().size() > 2,
        exceptions::GenericError,
        "MM supports only 1 & 2 dim Tensors atm");

    check_op_generic(lhs, rhs);

    // if not dot product
    if (lhs.shape() != rhs.shape() || lhs.shape().size() != 1) {
        CPPGRAD_CHECK_EQ(lhs.shape()[1], rhs.shape()[0],
            exceptions::GenericError,
            "LHS cols size not eq to RHS rows");

        auto out = Tensor::create_dirty({ lhs.shape()[0], rhs.shape()[1] }, lhs.dtype(), lhs.get_align(), lhs.device().clone());
        return MatmulOp::apply({ out, lhs, rhs })[0];
    } else {
        check_op_elementwise(lhs, rhs);
        auto out = Tensor::create_dirty({ 1 }, lhs.dtype(), lhs.get_align(), lhs.device().clone());
        return DotProductOp::apply({ out, lhs, rhs })[0];
    }
}

}