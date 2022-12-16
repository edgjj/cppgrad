#ifndef CPPGRAD_TENSOR_BINARY_OPS_HPP
#define CPPGRAD_TENSOR_BINARY_OPS_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"

namespace cppgrad {

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor pow(const Tensor& lhs, const Tensor& rhs);
Tensor mm(const Tensor& lhs, const Tensor& rhs);

}

#endif