#include "cppgrad/autograd/context.hpp"
#include "cppgrad/autograd/grad_mode.hpp"
#include "cppgrad/autograd/node.hpp"
#include "cppgrad/itertools/itertools.hpp"
#include "cppgrad/tensor/tensor.hpp"
#include <iostream>
#include <unordered_set>

namespace cppgrad::autograd {

struct AutogradContext : AutogradInterface {
    AutogradContext()
    {
        // acquire thread-local grad mode
        _requires_grad = ThreadLocalGradState::get();
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
        return _requires_grad || _grad_fn;
    }

private:
    bool _requires_grad { true };
    Tensor _grad;
    std::shared_ptr<Node> _grad_fn; // can be shared between multiple Tensors
};


namespace impl {

std::unique_ptr<AutogradInterface> AutogradContextFactory::make()
{
    return std::make_unique<AutogradContext>();
}

const Tensor& AutogradContextFactory::empty_tensor()
{
    static Tensor _empty;
    return _empty;
}

    using tensor_list = std::vector<Tensor>;
    using tensor_hash_set = std::unordered_set<Tensor>;

    static void walk(Tensor& node, tensor_list& tensors, tensor_hash_set& visited)
    {
        visited.emplace(node);

        // check if tensor has grad_fn
        if (node.grad_fn()) {
            for (auto& p : node.grad_fn()->edges()) {
                if (visited.count(p) == 0) {
                    walk(p, tensors, visited);
                }
            }

            tensors.push_back(node);
        }
    }

    static tensor_list walk(Tensor& root)
    {
        tensor_hash_set visited;
        tensor_list tensors;

        walk(root, tensors, visited);

        return std::move(tensors);
    }

}

void backward(Tensor& root)
{
    // no op
    if (!root.requires_grad()) {
        return;
    }

    auto topo = impl::walk(root);
    auto any_requires_grad = [](auto& edges) {
        for (auto& i : edges) {
            if (i.requires_grad()) {
                return true;
            }
        }

        return false;
    };

    // init root grad Tensor
    root.grad() = Tensor::create_dirty(root.shape(), root.dtype(), root.get_align(), root.device().clone());

    // this is ugliest thing ever; to be changed with autocast fill
    for_each_type(
        [&](auto tag) {
            using Type = decltype(tag);
            root.grad().fill(Type(1));
        },
        root.grad().dtype());

    // uh oh some crazy matches here
    for (auto& node : itertools::reversed(topo)) {

        auto& grad_fn = node.grad_fn();

        if (!any_requires_grad(grad_fn->edges())) {
            continue;
        }

        auto grads = grad_fn->backward(node.grad());

        for (auto [n, g] : itertools::combine(grad_fn->edges(), grads)) {
            if (!n.requires_grad() || g.empty()) {
                continue;
            }

            if (n.grad().empty()) {
                n.grad() = std::move(g);
            } else {
                // accumulate grad; g executor is actually same as n.grad() executor
                g.executor().sum(n.grad(), g, n.grad());
            }
        }

        // release grad_fn ownership for node
        grad_fn.reset();
    }

    // clear root grad tensor
    root.grad() = Tensor();
}

}