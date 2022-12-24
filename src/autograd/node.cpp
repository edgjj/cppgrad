#include "cppgrad/autograd/node.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::autograd {

namespace impl {

    tensor_list apply_finish(tensor_list& inputs, std::shared_ptr<Node>& node)
    {
        auto outputs = node->forward(std::move(inputs)); // consume inputs

        for (auto& i : outputs) {
            i.set_grad_fn(node);
        }

        return std::move(outputs);
    }
}

}