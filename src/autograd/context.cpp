#include "cppgrad/autograd/context.hpp"
#include "cppgrad/autograd/node.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::autograd {

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

}