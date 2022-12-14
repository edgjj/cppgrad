// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_NN_MODULE_HPP
#define CPPGRAD_NN_MODULE_HPP

#include "cppgrad/device/device.hpp"
#include "cppgrad/tensor/tensor_fwd.hpp"
#include <vector>

namespace cppgrad::nn {

using tensor_list = std::vector<Tensor>;
using tensor_ptr_list = std::vector<Tensor*>;

struct Module {

    virtual tensor_list forward(tensor_list inputs) = 0;

    template <typename... Tensors>
    tensor_list operator()(Tensors&&... tensors)
    {
        return forward({ std::forward<Tensors>(tensors)... });
    }

    tensor_list operator()(const tensor_list& inputs)
    {
        return forward(inputs);
    }

    void cpu();
    void cuda();

    void register_module(Module& new_child);

    virtual tensor_ptr_list get_parameters();
    virtual ~Module();

private:
    std::vector<Module*> _child;
    Module* _parent;
};

}

#endif