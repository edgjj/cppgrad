// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_NN_LOSS_FN_HPP
#define CPPGRAD_NN_LOSS_FN_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"

namespace cppgrad::nn {

Tensor mse_loss(const Tensor& y_hat, const Tensor& y);

}

#endif