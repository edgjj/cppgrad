#include "cppgrad/tensor/ops/grad_ops.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad {

// AddOp
tensor_list AddOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    x.executor().sum(x, y, x);

    return { x };
}

tensor_list AddOp::backward(const Tensor& prev_grad)
{
    auto grad = prev_grad.clone();
    return { grad, grad };
}

// SubOp;
tensor_list SubOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    x.executor().sub(x, y, x);

    return { x };
}

tensor_list SubOp::backward(const Tensor& prev_grad)
{
    // same as AddOp
    auto grad = prev_grad.clone();
    // multiply grad_y by -1
    return { grad, grad };
}

// MultiplyOp
tensor_list MultiplyOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    save_for_backward(x, y);
    x.executor().mul(x, y, x);

    return { x };
}

tensor_list MultiplyOp::backward(const Tensor& prev_grad)
{
    auto &x = saved()[0],
         &y = saved()[1];

    auto grad_x = x.clone(),
         grad_y = y.clone();

    grad_x.executor().mul(y, prev_grad, grad_x);
    grad_y.executor().mul(x, prev_grad, grad_y);

    return { grad_x, grad_y };
}

// DivisionOp
tensor_list DivisionOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    save_for_backward(x, y);
    x.executor().div(x, y, x);

    return { x };
}

tensor_list DivisionOp::backward(const Tensor& prev_grad)
{
    auto &x = saved()[0],
         &y = saved()[1];

    auto grad_x = x.clone(),
         grad_y = y.clone();

    // (y*grad - x * grad) / y^2
    grad_x.executor().mul(y, prev_grad, grad_x);
    grad_y.executor().mul(x, prev_grad, grad_y);

    // pow 2
    y.executor().mul(y, y, y);
    grad_x.executor().div(grad_x, y, grad_x);
    grad_y.executor().div(grad_y, y, grad_y);
    // missing sub for grad_y / mul -1 there
    return { grad_x, grad_y };
}

// PowOp
tensor_list PowOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    // save inputs
    save_for_backward(x, y);
    x.executor().pow(x, y, x);

    // save output
    save_for_backward(x);

    return { x };
}

tensor_list PowOp::backward(const Tensor& prev_grad)
{
    auto &x = saved()[0],
         &y = saved()[1],
         &out = saved()[2];

    auto grad_x = out.clone(),
         grad_y = out.clone();

    grad_x.executor().mul(prev_grad, grad_x, grad_x); // out->grad * pow(self->val, other->val)
    y.executor().div(y, x, y); // other->val = other->val / self->val
    // out->grad * other->val * pow(self->val, other->val) / self->val -> out->grad * other->val * pow(self->val, other->val - 1)
    grad_x.executor().mul(grad_x, y, grad_x);

    grad_y.executor().mul(prev_grad, grad_y, grad_y); // out->grad * pow(self->val, other->val)
    x.executor().log(x, x); // x = ln(x)
    grad_y.executor().mul(x, grad_y, grad_y); // out->grad * pow(self->val, other->val) * ln(x)

    return { grad_x, grad_y };
}

// MatmulOp
tensor_list MatmulOp::forward(tensor_list inputs)
{
    auto &out = inputs[0],
         &x = inputs[1],
         &y = inputs[2];

    out.executor().matmul(x, y, out);
    save_for_backward(x, y);

    return { out };
}

tensor_list MatmulOp::backward(const Tensor& prev_grad)
{
    auto &x = saved()[0],
         &y = saved()[1];

    auto grad_x = x.clone(),
         grad_y = y.clone();

    grad_x.executor().matmul(prev_grad, y.T(), grad_x); // grad_x = g @ y_T
    grad_y.executor().matmul(x.T(), prev_grad, grad_y); // grad_y = x_T @ g

    return { grad_y, grad_y };
}

// DotProductOp
tensor_list DotProductOp::forward(tensor_list inputs)
{
    auto &out = inputs[0],
         &x = inputs[1],
         &y = inputs[2];

    out.executor().dot(x, y, out);
    save_for_backward(x, y);

    return { out };
}

tensor_list DotProductOp::backward(const Tensor& prev_grad)
{
    auto &x = saved()[0],
         &y = saved()[1];

    auto grad_x = x.clone(),
         grad_y = y.clone();

    // using loop() fakes out Tensor length, allowing iteration point to same mem location
    grad_x.executor().mul(y, prev_grad.loop(y.shape()), grad_x);
    grad_y.executor().mul(x, prev_grad.loop(y.shape()), grad_y);

    return { grad_x, grad_y };
    return { grad_x, grad_y };
}

}