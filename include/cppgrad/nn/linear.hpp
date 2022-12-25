#ifndef CPPGRAD_NN_LINEAR_HPP
#define CPPGRAD_NN_LINEAR_HPP

#include "cppgrad/nn/module.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad::nn {

struct Linear : Module {

    Linear(size_t in_size, size_t out_size, bool needs_bias = true);

    tensor_list forward(tensor_list inputs) override;
    tensor_ptr_list get_parameters() override;

private:
    Tensor _w;
    Tensor _b;
};

}

#endif