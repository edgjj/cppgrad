#ifndef CPPGRAD_AUTOGRAD_CONTEXT_HPP
#define CPPGRAD_AUTOGRAD_CONTEXT_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"
#include <memory>

namespace cppgrad::autograd {

struct Node;

struct AutogradInterface {
    /**
     * @brief Retrieves mutable gradient Tensor
     *
     * @return Tensor&
     */
    virtual Tensor& grad() = 0;

    /**
     * @brief Retrieves immutable gradient Tensor
     *
     * @return const Tensor&
     */
    virtual const Tensor& grad() const = 0;

    /**
     * @brief Sets the grad_fn Node object
     *
     * @param new_grad_fn
     */
    virtual void set_grad_fn(std::shared_ptr<Node> new_grad_fn) = 0;

    /**
     * @brief Retrieves grad_fn Node object
     *
     */
    virtual std::shared_ptr<Node>& grad_fn() = 0;

    /**
     * @brief Sets whether derived requires grad or not
     *
     * @param new_requires_grad
     */
    virtual void set_requires_grad(bool new_requires_grad) = 0;

    /**
     * @brief Retrieves if derived requires grad or not
     *
     */
    virtual bool requires_grad() const = 0;

    virtual ~AutogradInterface() = default;
};

struct AutogradContextFactory {
    static std::unique_ptr<AutogradInterface> make();
    static const Tensor& empty_tensor();
};

namespace impl {

    /**
     * @brief Backward passes starting from root Tensor
     *
     * @param root
     */
    void backward(Tensor& root);

}

}

#endif