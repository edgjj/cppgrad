#ifndef CPPGRAD_AUTOGRAD_NODE_HPP
#define CPPGRAD_AUTOGRAD_NODE_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"
#include <memory>
#include <utility>
#include <vector>

namespace cppgrad::autograd {

using tensor_list = std::vector<Tensor>;

struct Node {
    virtual tensor_list forward(tensor_list inputs) = 0;
    virtual tensor_list backward(tensor_list inputs) = 0;

    tensor_list apply(tensor_list inputs)
    {
        auto result = forward(std::move(inputs));
        // auto grad_fn =
        // for (auto& i : result) {
        //     // assign parents
        // }
    }

    void save_for_backward(const Tensor& variable)
    {
        _saved_data.push_back(variable);
    }

    void set_edges(tensor_list input_edges)
    {
        _edges = std::move(input_edges);
    }

    virtual ~Node() = default;

private:
    tensor_list _saved_data;
    tensor_list _edges;
};

}

#endif