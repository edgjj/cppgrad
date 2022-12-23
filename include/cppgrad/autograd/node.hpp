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

namespace impl {
    // since apply() needs complete Tensoor, we do this
    void apply_finish(tensor_list& outputs, std::shared_ptr<Node>& node);
}

template <typename Fn>
struct CustomNode {
    static tensor_list apply(tensor_list inputs)
    {
        std::shared_ptr<Node> op { new Fn() };

        op->set_edges(inputs);
        auto outputs = op->forward(std::move(inputs)); // consume inputs

        impl::apply_finish(outputs, op);
        return std::move(outputs);
    }
};

}

#endif