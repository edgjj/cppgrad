#include "cppgrad/nn/module.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::nn {

Module::~Module() = default;

tensor_ptr_list Module::get_parameters()
{
    tensor_ptr_list parameter_list;
    for (auto& i : _child) {
        auto child_parameters = i->get_parameters();

        parameter_list.insert(parameter_list.end(),
            child_parameters.begin(),
            child_parameters.end());
    }

    return std::move(parameter_list);
}

void Module::register_module(Module& new_child)
{
    _child.push_back(&new_child);
}

void Module::cpu()
{
    for (auto& i : get_parameters()) {
        *i = (*i).cpu();
    }
}

void Module::cuda()
{
    for (auto& i : get_parameters()) {
        *i = (*i).cuda();
    }
}

}