#include "cppgrad/tensor/ops/grad_ops.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad {

// AddOp
tensor_list AddOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    auto out = x.clone();
    out.executor().add(out, y, out);

    return { out };
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

    auto out = x.clone();
    out.executor().sub(out, y, out);

    return { out };
}

tensor_list SubOp::backward(const Tensor& prev_grad)
{
    auto grad_x = prev_grad.clone(),
         grad_y = prev_grad.clone();

    grad_y.executor().neg(grad_y, grad_y); // negate Y grad
    return { grad_x, grad_y };
}

// MultiplyOp
tensor_list MultiplyOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    auto out = x.clone();

    save_for_backward(x, y);
    out.executor().mul(out, y, out);

    return { out };
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

    auto out = x.clone();

    save_for_backward(x, y);
    out.executor().div(out, y, out);

    return { out };
}

tensor_list DivisionOp::backward(const Tensor& prev_grad)
{
    auto &x = saved()[0],
         &y = saved()[1];

    auto grad_x = x.clone(),
         grad_y = y.clone();

    // (y*grad - x * grad) / y^2
    grad_x.executor().mul(prev_grad, y, grad_x); // f'(x) * y
    grad_y.executor().mul(prev_grad, x, grad_y); // g'(x) * x

    // pow 2
    y.executor().mul(y, y, y);
    grad_x.executor().div(grad_x, y, grad_x);
    grad_y.executor().div(grad_y, y, grad_y);

    grad_y.executor().neg(grad_y, grad_y);

    return { grad_x, grad_y };
}

// PowOp
tensor_list PowOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    auto out = x.clone();
    out.executor().pow(out, y, out);

    // save inputs & output
    save_for_backward(x, y, out);

    return { out };
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
    auto &x = inputs[0],
         &y = inputs[1];

    auto out = Tensor::create_dirty({ x.shape()[0], y.shape()[1] }, x.dtype(), x.get_align(), x.device().clone());

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

    return { grad_x, grad_y };
}

// DotProductOp
tensor_list DotProductOp::forward(tensor_list inputs)
{
    auto &x = inputs[0],
         &y = inputs[1];

    auto out = Tensor::create_dirty({ 1 }, x.dtype(), x.get_align(), x.device().clone());
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
}

// LogOp
tensor_list LogOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    auto out = x.clone();

    save_for_backward(x);
    out.executor().log(out, out);

    return { out };
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
    auto out = x.clone();

    out.executor().exp(out, out);
    save_for_backward(out);

    return { out };
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
    auto out = x.clone();

    out.executor().relu(out, out);
    save_for_backward(out);

    return { out };
}

tensor_list ReluOp::backward(const Tensor& prev_grad)
{
    auto& out = saved()[0];
    auto grad = prev_grad.clone();

    grad.executor().sign(out, grad);
    grad.executor().mul(prev_grad, grad, grad);

    return { grad }; // (x > 0) * prev_grad -> sign(out) * prev_grad
}

// TanhOp
tensor_list TanhOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    auto out = x.clone();

    out.executor().tanh(out, out);
    save_for_backward(out);

    return { out };
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

// SigmoidOp; THIS LITTLE BOY SCREWES GRADIENT UP!!! incorrect somewhere
tensor_list SigmoidOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    auto out = x.clone();
    auto one = Tensor(1, out.dtype()).loop(out.shape());
    one = out.is_cuda_tensor() ? one.cuda() : one;

    out.executor().neg(out, out);
    out.executor().exp(out, out);

    out.executor().add(one, out, out);
    out.executor().div(one, out, out);

    save_for_backward(out);

    return { out };
}

tensor_list SigmoidOp::backward(const Tensor& prev_grad)
{
    auto& out = saved()[0];
    auto grad = prev_grad.clone(); // 1 - tanh(x) * tanh(x)

    grad.executor().mul(out, out, grad); // sigma_x^2
    grad.executor().sub(out, grad, grad); // sigma_x - sigma_x^2 -> sigma_x(1 - sigma_x);

    return { grad };
}

// SignOp
tensor_list SignOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    auto out = x.clone();

    out.executor().sign(out, out);

    return { out };
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
    auto out = x.clone();

    out.executor().neg(out, out);

    return { out };
}

tensor_list NegOp::backward(const Tensor& prev_grad)
{
    auto grad = prev_grad.clone();
    grad.executor().neg(prev_grad, grad); // negate grad

    return { grad };
}

// SumOp
tensor_list SumOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    _saved_shape = x.shape();

    auto out = Tensor::create_dirty({ 1 }, x.dtype(), x.get_align(), x.device().clone());
    out.executor().sum(x, out);

    return { out };
}

tensor_list SumOp::backward(const Tensor& prev_grad)
{
    auto grad = prev_grad.clone().loop(_saved_shape);
    return { grad };
}

PermuteOp::PermuteOp(std::vector<size_t> order)
{
    _saved_order = std::move(order);
}

tensor_list PermuteOp::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    auto out = x.clone();

    for (size_t k = 0; k < _saved_order.size(); k++) {
        out._storage->_shape[k] = x._storage->_shape[_saved_order[k]];
        out._storage->_strides[k] = x._storage->_strides[_saved_order[k]];
    }

    return { out };
}

tensor_list PermuteOp::backward(const Tensor& prev_grad)
{
    auto grad = prev_grad.clone();
    std::vector<size_t> new_order(_saved_order.size());

    std::iota(new_order.begin(), new_order.end(), 0);
    std::sort(new_order.begin(), new_order.end(), [&](size_t a, size_t b) {
        return _saved_order[a] < _saved_order[b];
    });

    for (size_t k = 0; k < new_order.size(); k++) {
        grad._storage->_shape[k] = prev_grad._storage->_shape[new_order[k]];
        grad._storage->_strides[k] = prev_grad._storage->_strides[new_order[k]];
    }

    return { grad };
}

}