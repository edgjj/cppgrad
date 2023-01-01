// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "cppgrad/nn/losses.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::nn {

Tensor mse_loss(const Tensor& y_hat, const Tensor& y)
{
    // y_hat is most likely output from NN
    auto loss = y_hat - y;
    loss = loss * loss;

    auto divisor = Tensor::full({ 1 }, y.numel(), loss.dtype(), loss.device());

    loss = sum(loss) / divisor;
    return loss;
}

}