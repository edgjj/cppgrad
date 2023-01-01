// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <cppgrad/autograd/node.hpp>
#include <cppgrad/cppgrad.hpp>

#include <iostream>

using namespace cppgrad;

/*
    This is an example of custom ops, suitable for Autograd engine.

    For simplicity, these ops clone their input, but
    real code may have different behavior for grad/non-grad inputs.

    Typical Op implementation should be splitted into cpp/hpp,
    where cpp also has Tensor include.

    Supplying Op type as template parameter when inheriting CustomNode
    is required for apply to work.

*/

struct AddOp : autograd::CustomNode<AddOp> {
    tensor_list forward(tensor_list inputs) override
    {
        auto &x = inputs[0],
             &y = inputs[1];

        auto out = x.clone();
        x.executor().add(x, y, out);

        return { out };
    }

    tensor_list backward(const Tensor& prev_grad) override
    {
        auto grad = prev_grad.clone(); // need to solve this issue too
        return { grad, grad };
    }
};

struct MulOp : autograd::CustomNode<MulOp> {
    tensor_list forward(tensor_list inputs) override
    {
        auto &x = inputs[0],
             &y = inputs[1];

        auto out = x.clone();

        save_for_backward(x, y);
        x.executor().mul(x, y, out);

        return { out };
    }

    tensor_list backward(const Tensor& prev_grad) override
    {
        auto &x = saved()[0],
             &y = saved()[1];

        auto grad_x = x.clone(),
             grad_y = y.clone();

        grad_x.executor().mul(y, prev_grad, grad_x);
        grad_y.executor().mul(x, prev_grad, grad_y);

        return { grad_x, grad_y };
    }
};

int main()
{
    autograd::ForceGradGuard guard;

    auto t1 = Tensor(2.0);
    auto t2 = Tensor(4.0);

    auto a1 = AddOp::apply({ t1, t2 })[0];
    auto m1 = MulOp::apply({ a1, t1 })[0];
    auto a2 = AddOp::apply({ m1, t2 })[0];
    auto m2 = MulOp::apply({ t1, a2 })[0];
    auto a3 = AddOp::apply({ m2, t2 })[0];

    auto t3 = a3;
    t3.backward();

    std::cout << "c value: " << t3.item<f64>() << "; \n"
              << "a value: " << t1.item<f64>() << "; grad(): " << t1.grad().item<f64>() << "; \n"
              << "b value: " << t2.item<f64>() << "; grad(): " << t2.grad().item<f64>() << ";" << std::endl;

    return 0;
}