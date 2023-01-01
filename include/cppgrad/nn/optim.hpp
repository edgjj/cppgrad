// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_NN_OPTIM_HPP
#define CPPGRAD_NN_OPTIM_HPP

namespace cppgrad::nn::optim {

template <typename Net>
struct SGD {
    SGD(Net& net, double learning_rate)
        : _lr(learning_rate)
    {
        _params = net.get_parameters();
    }

    void step()
    {
        for (auto& i : _params) {
            auto lr_tensor = Tensor::full(i->grad().shape(), _lr, i->dtype(), i->device());
            (*i) -= (i->grad() * lr_tensor);

            // restore state
            i->set_requires_grad(true);
        }
    }

    void zero_grad()
    {
        // sets grads to nothing
        for (auto& i : _params) {
            (*i) = (*i); // this efficiently washes grad
        }
    }

private:
    nn::tensor_ptr_list _params;
    double _lr { 0.0 };
};

}

#endif