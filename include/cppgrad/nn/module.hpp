#ifndef CPPGRAD_NN_MODULE_HPP
#define CPPGRAD_NN_MODULE_HPP

#include "cppgrad/device/device.hpp"
#include "cppgrad/tensor/tensor_fwd.hpp"
#include <unordered_set>

namespace cppgrad::nn {

using tensor_list = std::vector<Tensor>;
using tensor_ptr_list = std::vector<Tensor*>;

struct Module {

    virtual tensor_list forward(tensor_list inputs) = 0;
    tensor_list operator()(tensor_list inputs);

    void cpu();
    void cuda();

    void register_module(Module& new_child);

    virtual tensor_ptr_list get_parameters();
    virtual ~Module();

private:
    std::unordered_set<Module*> _child;
    Module* _parent;
};

}

#endif