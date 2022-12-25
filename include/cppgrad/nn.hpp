#ifndef CPPGRAD_NN_HPP
#define CPPGRAD_NN_HPP

#include <vector>

namespace cppgrad {

namespace nn {

    /*
            atm just a concept
    */

    // look at this: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module

    // also this: https://github.com/pytorch/pytorch/blob/04108592a362848a5d3af4332f7628a14e312174/c10/core/GradMode.h
    class Module {
    public:
        Module(Module* parent = nullptr)
            : _parent(parent)
        {
            if (_parent) {
                _parent->register_child(this);
                // register itself
                // like _parent->register_child()
            }
        }

        virtual void get_parameters()
        {
            return; // return nothing in case we dont have params
        }

    private:
        void register_child(Module* new_child)
        {
            _child.push_back(new_child);
        }

        std::vector<Module*> _child;
        Module* _parent; // no ownership; btw could be made better thru class-local shit
    };
}

}

#endif