#include "cppgrad/autograd/context.hpp"
#include "cppgrad/autograd/node.hpp"
#include "cppgrad/itertools/itertools.hpp"
#include "cppgrad/tensor/tensor.hpp"
#include <set>

namespace cppgrad::autograd {

using tensor_ptr_list = std::vector<Tensor*>;
using tensor_ptr_set = std::set<Tensor*>;

struct AutogradContext : AutogradInterface {
    AutogradContext()
    {
        // we can access some thread-local storage there to automatically set _requires_grad
        // NoGradGuard moment
    }

    Tensor& grad() override
    {
        return _grad;
    };

    const Tensor& grad() const override
    {
        return _grad;
    }

    void set_grad_fn(std::shared_ptr<Node> new_grad_fn) override
    {
        _grad_fn = std::move(new_grad_fn);
    }

    std::shared_ptr<Node>& grad_fn() override
    {
        return _grad_fn;
    }

    void set_requires_grad(bool new_requires_grad) override
    {
        _requires_grad = new_requires_grad;
    }

    bool requires_grad() const override
    {
        return _requires_grad;
    }

private:
    bool _requires_grad { true };
    Tensor _grad;
    std::shared_ptr<Node> _grad_fn; // can be shared between multiple Tensors
};

std::unique_ptr<AutogradInterface> AutogradContextFactory::make()
{
    return std::make_unique<AutogradContext>();
}

const Tensor& AutogradContextFactory::empty_tensor()
{
    static Tensor _empty;
    return _empty;
}

namespace impl {
    static void walk(Tensor& node, tensor_ptr_list& tensors, tensor_ptr_set& visited)
    {
        visited.emplace(&node);

        for (auto& p : node.grad_fn()->edges()) {
            if (visited.count(&p) == 0) {
                walk(p, tensors, visited);
            }
        }

        tensors.push_back(&node);
    }

    static tensor_ptr_list walk(Tensor& root)
    {
        tensor_ptr_set visited;
        tensor_ptr_list tensors;

        walk(root, tensors, visited);

        return std::move(tensors);
    }

    void backward(Tensor& root)
    {
        auto topo = walk(root);
        auto any_requires = [](auto& edges) {
            for (auto& i : edges) {
                if (!i.requires_grad()) {
                    return false;
                }
            }

            return true;
        };

        // init root grad Tensor
        root.grad() = Tensor::create_dirty(root.shape(), root.dtype(), root.get_align(), root.device().clone());

        // this is ugliest thing ever; to be changed with autocast fill
        for_each_type(
            [&](auto tag) {
                using Type = decltype(tag);
                root.fill(Type(1));
            },
            root.dtype());

        // uh oh some crazy matches here
        for (auto& i : itertools::reversed(topo)) {
            auto& node = *i;

            auto& grad_fn = node.grad_fn();

            if (!any_requires(grad_fn->edges())) {
                continue;
            }

            auto grads = grad_fn->backward(node.grad());

            for (auto [n, g] : itertools::combine(grad_fn->edges(), grads)) {
                if (n.requires_grad() && !g.empty()) {
                    n.grad() = std::move(g);
                }
            }

            // release grad_fn ownership for node
            grad_fn.reset();
        }
    }

}

}