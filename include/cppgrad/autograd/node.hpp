#ifndef CPPGRAD_AUTOGRAD_NODE_HPP
#define CPPGRAD_AUTOGRAD_NODE_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace cppgrad {

using tensor_list = std::vector<Tensor>;

namespace autograd {

    struct Node {
        virtual tensor_list forward(tensor_list inputs) = 0;
        virtual tensor_list backward(const Tensor& grad) = 0;

        void set_edges(tensor_list input_edges)
        {
            _edges = std::move(input_edges);
        }

        tensor_list& edges()
        {
            return _edges;
        }

        virtual ~Node() = default;

    protected:
        template <typename... Tensors>
        void save_for_backward(Tensors&&... variables)
        {
            _saved_data.insert(_saved_data.end(), { variables.clone()... });
        }

        tensor_list& saved()
        {
            return _saved_data;
        }

    private:
        tensor_list _saved_data;
        tensor_list _edges;
    };

    namespace impl {
        // since apply() needs complete Tensoor, we do this
        tensor_list apply_finish(tensor_list& outputs, std::shared_ptr<Node>& node);
    }

    template <typename Fn>
    struct CustomNode : Node {
        static tensor_list apply(tensor_list inputs)
        {
            // check if at least 1 input requires grad; if not - just make pure forward call
            if (std::find_if(inputs.begin(), inputs.end(), [](auto& t) { return t.requires_grad(); }) == inputs.end()) {
                return Fn {}.forward(std::move(inputs));
            }

            std::shared_ptr<Node> op { new Fn() };

            op->set_edges(inputs);
            return impl::apply_finish(inputs, op);
        }
    };
}

}

#endif