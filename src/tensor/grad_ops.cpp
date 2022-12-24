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
    auto grad_x = prev_grad.clone(),
         grad_y = prev_grad.clone();

    grad_y.executor().neg(grad_y, grad_y); // negate Y grad
    return { grad_x, grad_y };
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

    grad_x.executor().mul(prev_grad, y, grad_x);
    grad_y.executor().mul(prev_grad, x, grad_y);

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
    grad_x.executor().mul(prev_grad, y, grad_x);
    grad_y.executor().mul(prev_grad, x, grad_y);

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

// LogOp
tensor_list LogOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];

    save_for_backward(x);
    x.executor().log(x, x);

    return { x };
}

tensor_list LogOp::backward(const Tensor& prev_grad)
{
    auto& x = saved()[0];

    auto grad = prev_grad.clone();
    grad.executor().div(grad, x, grad); // d * 1 / x

    return { grad };
}

// ExpOp
tensor_list ExpOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];

    x.executor().exp(x, x);
    save_for_backward(x);

    return { x };
}

tensor_list ExpOp::backward(const Tensor& prev_grad)
{
    auto& out = saved()[0];
    auto grad = prev_grad.clone();

    grad.executor().mul(grad, out, grad); // d * e ^ x

    return { grad };
}

// ReluOp
tensor_list ReluOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];

    x.executor().relu(x, x);
    save_for_backward(x);

    return { x };
}

tensor_list ReluOp::backward(const Tensor& prev_grad)
{
    auto& out = saved()[0];
    auto grad = prev_grad.clone();

    out.executor().sign(out, out);
    grad.executor().mul(out, grad, grad);

    return { grad }; // (x > 0) * prev_grad -> sign(out) * prev_grad
}

// TanhOp
tensor_list TanhOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];

    x.executor().tanh(x, x);
    save_for_backward(x);

    return { x };
}

tensor_list TanhOp::backward(const Tensor& prev_grad)
{
    auto& x = saved()[0];
    auto grad = prev_grad.clone(); // 1 - tanh(x) * tanh(x)

    x.executor().mul(x, x, x); // tanh^2(x)
    x.executor().mul(x, grad, x); // tanh^2(x) * g

    grad.executor().sub(grad, x, grad); // g - g * tanh^2(x)

    return { grad };
}

// SignOp
tensor_list SignOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    x.executor().sign(x, x);

    return { x };
}

tensor_list SignOp::backward(const Tensor& prev_grad)
{
    auto grad = prev_grad.clone();
    grad.executor().sub(prev_grad, prev_grad, grad); // zero grad;

    return { grad };
}

// NegOp
tensor_list NegOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    x.executor().neg(x, x);

    return { x };
}

tensor_list NegOp::backward(const Tensor& prev_grad)
{
    auto grad = prev_grad.clone();
    grad.executor().neg(prev_grad, grad); // negate grad

    return { grad };
}

}