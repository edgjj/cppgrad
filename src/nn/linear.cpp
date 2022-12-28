#include "cppgrad/nn/linear.hpp"
#include "cppgrad/exceptions/generic_error.hpp"
#include <iostream>

namespace cppgrad::nn {

Linear::Linear(size_t in_size, size_t out_size, DType dtype, bool needs_bias)
{
    if (in_size == 0 || out_size == 0) {
        throw exceptions::GenericError("Invalid sizes specified for Linear layer.");
    }

    _w = Tensor::rand({ out_size, in_size }, -0.1, 0.1, dtype, kCPU);
    _w.set_requires_grad(true);

    if (needs_bias) {
        _b = Tensor::rand({ out_size }, -0.1, 0.1, dtype, kCPU);
        _b.set_requires_grad(true);
    }
}

tensor_list Linear::forward(tensor_list inputs)
{
    auto& x = inputs[0];
    auto y_hat = cppgrad::mm(x, _w.T());

    if (!_b.empty()) {
        y_hat = y_hat + _b;
    }

    return { y_hat };
}

tensor_ptr_list Linear::get_parameters()
{
    tensor_ptr_list params { &_w };
    // check if there's bias
    if (!_b.empty()) {
        params.push_back(&_b);
    }

    return params;
}

}