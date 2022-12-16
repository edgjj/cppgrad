#ifndef CPPGRAD_TENSOR_UNARY_OPS_HPP
#define CPPGRAD_TENSOR_UNARY_OPS_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"

namespace cppgrad {

Tensor& operator+=(Tensor& lhs, const Tensor& rhs);
Tensor& operator-=(Tensor& lhs, const Tensor& rhs);
Tensor& operator*=(Tensor& lhs, const Tensor& rhs);

}

#endif