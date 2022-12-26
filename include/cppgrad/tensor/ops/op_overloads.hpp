#ifndef CPPGRAD_TENSOR_OP_OVERLOADS_HPP
#define CPPGRAD_TENSOR_OP_OVERLOADS_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"

namespace cppgrad {

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs);

Tensor pow(const Tensor& lhs, const Tensor& rhs);
Tensor mm(const Tensor& lhs, const Tensor& rhs);
Tensor sum(const Tensor& lhs);

Tensor& operator+=(Tensor& lhs, const Tensor& rhs);
Tensor& operator-=(Tensor& lhs, const Tensor& rhs);
Tensor& operator*=(Tensor& lhs, const Tensor& rhs);
Tensor& operator/=(Tensor& lhs, const Tensor& rhs);

Tensor log(const Tensor& lhs);
Tensor exp(const Tensor& lhs);
Tensor relu(const Tensor& lhs);
Tensor tanh(const Tensor& lhs);
Tensor sigmoid(const Tensor& lhs);
Tensor sign(const Tensor& lhs);
Tensor neg(const Tensor& lhs);

}

#endif