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

    lhs = AddOp::apply({ lhs, rhs })[0];
    return lhs;
}

Tensor& operator-=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    lhs = SubOp::apply({ lhs, rhs })[0];
    return lhs;
}

Tensor& operator*=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    lhs = MultiplyOp::apply({ lhs, rhs })[0];
    return lhs;
}

Tensor& operator/=(Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    lhs = DivisionOp::apply({ lhs, rhs })[0];
    return lhs;
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return AddOp::apply({ lhs, rhs })[0];
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return SubOp::apply({ lhs, rhs })[0];
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return MultiplyOp::apply({ lhs, rhs })[0];
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return DivisionOp::apply({ lhs, rhs })[0];
}

Tensor operator-(const Tensor& lhs)
{
    return neg(lhs);
}

Tensor pow(const Tensor& lhs, const Tensor& rhs)
{
    check_op_generic(lhs, rhs);
    check_op_elementwise(lhs, rhs);

    return PowOp::apply({ lhs, rhs })[0];
}

Tensor mm(const Tensor& lhs, const Tensor& rhs)
{
    CPPGRAD_CHECK_FALSE(lhs.shape().size() > 2 && rhs.shape().size() > 2,
        exceptions::GenericError,
        "MM supports only 1 & 2 dim Tensors atm");

    check_op_generic(lhs, rhs);

    // if not dot product
    if (lhs.shape() != rhs.shape() || lhs.shape().size() != 1) {
        CPPGRAD_CHECK_FALSE(lhs.shape().size() != 2 || rhs.shape().size() != 2,
            exceptions::GenericError,
            "Matmul requires both Tensors be 2-dim");

        CPPGRAD_CHECK_EQ(lhs.shape()[1], rhs.shape()[0],
            exceptions::GenericError,
            "LHS cols size not eq to RHS rows");

        return MatmulOp::apply({ lhs, rhs })[0];
    } else {
        check_op_elementwise(lhs, rhs);
        return DotProductOp::apply({ lhs, rhs })[0];
    }
}

Tensor sum(const Tensor& lhs)
{
    return SumOp::apply({ lhs })[0];
}

// we need to solve that const incorrectness thing;
// shallow copying const tensor allows to modifying parent tensor contents
// but i think it's not possible without killing whole shallow copy thing

Tensor log(const Tensor& lhs)
{
    return LogOp::apply({ lhs })[0];
}

Tensor exp(const Tensor& lhs)
{
    return ExpOp::apply({ lhs })[0];
}

Tensor relu(const Tensor& lhs)
{
    return ReluOp::apply({ lhs })[0];
}

Tensor tanh(const Tensor& lhs)
{
    return TanhOp::apply({ lhs })[0];
}

// synthesize from exp
Tensor sigmoid(const Tensor& lhs)
{
    auto one = Tensor::create_dirty(lhs.shape(), lhs.dtype(), 8, lhs.device().clone());
    one.fill(1.0);

    // one MUST be rhs; due to it's loop nature

    auto numerator = exp(lhs);
    auto denominator = numerator + one;

    return numerator / denominator;
}

Tensor sign(const Tensor& lhs)
{
    return SignOp::apply({ lhs })[0];
}

Tensor neg(const Tensor& lhs)
{
    return NegOp::apply({ lhs })[0];
}
}