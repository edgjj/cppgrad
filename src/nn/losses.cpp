#include "cppgrad/nn/losses.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::nn {

Tensor mse_loss(const Tensor& y_hat, const Tensor& y)
{
    // y_hat is most likely output from NN
    auto loss = y_hat - y;
    loss = loss * loss;

    auto divisor = Tensor(y.numel(), loss.dtype());
    if (loss.is_cuda_tensor()) { // to be changed; we definitely need to lift scalars without allocation
        divisor = divisor.cuda();
    }

    loss = sum(loss) / divisor;
    return loss;
}

}