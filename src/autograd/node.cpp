#include "cppgrad/autograd/node.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::autograd {

namespace impl {

    void apply_finish(tensor_list& outputs, std::shared_ptr<Node>& node)
    {
        for (auto& i : outputs) {
            i.set_grad_fn(node);
        }
    }

}

}