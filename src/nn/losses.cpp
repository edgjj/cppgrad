#include "cppgrad/nn/losses.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::nn {

Tensor mse_loss(const Tensor& y_hat, const Tensor& y)
{
    // y_hat is most likely output from NN
    auto loss = y_hat - y;
    loss = loss * loss;

    auto divisor = Tensor::create_dirty({ 1 }, loss.dtype(), 8, loss.device().clone());
    divisor.fill(y.numel());

    loss = sum(loss) / divisor;
    return loss;
}

}