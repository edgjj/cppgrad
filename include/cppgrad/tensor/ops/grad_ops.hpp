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

struct LogOp : autograd::CustomNode<LogOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct ExpOp : autograd::CustomNode<ExpOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct ReluOp : autograd::CustomNode<ReluOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct TanhOp : autograd::CustomNode<TanhOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct SigmoidOp : autograd::CustomNode<SigmoidOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct SignOp : autograd::CustomNode<SignOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct NegOp : autograd::CustomNode<NegOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;
};

struct SumOp : autograd::CustomNode<SumOp> {
    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;

private:
    std::vector<size_t> _saved_shape;
};

struct PermuteOp : autograd::CustomNode<PermuteOp> {
    PermuteOp(std::vector<size_t> order);

    tensor_list forward(tensor_list inputs) override;
    tensor_list backward(const Tensor& prev_grad) override;

private:
    std::vector<size_t> _saved_order;
};

}

#endif