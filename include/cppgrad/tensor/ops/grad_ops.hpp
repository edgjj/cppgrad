#ifndef CPPGRAD_TENSOR_GRAD_OPS_HPP
#define CPPGRAD_TENSOR_GRAD_OPS_HPP

#include "cppgrad/autograd/node.hpp"

namespace cppgrad {

struct AddOp : autograd::CustomNode<AddOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct SubOp : autograd::CustomNode<SubOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct MultiplyOp : autograd::CustomNode<MultiplyOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct DivisionOp : autograd::CustomNode<DivisionOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct PowOp : autograd::CustomNode<PowOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct MatmulOp : autograd::CustomNode<MatmulOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct DotProductOp : autograd::CustomNode<DotProductOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

}

#endif