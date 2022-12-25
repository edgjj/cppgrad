#ifndef CPPGRAD_NN_OPTIM_HPP
#define CPPGRAD_NN_OPTIM_HPP

namespace cppgrad::nn::optim {

template <typename Net>
struct SGD {
    SGD(Net& net, double learning_rate)
        : _lr(learning_rate)
    {
        _params = net.get_parameters();
    }

    void step()
    {
        for (auto& i : _params) {
            auto loop_tensor = i->is_cuda_tensor() ? Tensor(_lr, i->dtype()).cuda().loop(i->grad().shape())
                                                   : Tensor(_lr, i->dtype()).loop(i->grad().shape());

            (*i) -= (i->grad() * loop_tensor);

            // restore state
            i->set_requires_grad(true);
        }
    }

    void zero_grad()
    {
        // sets grads to nothing
        for (auto& i : _params) {
            i->grad() = Tensor();
        }
    }

private:
    nn::tensor_ptr_list _params;
    double _lr { 0.0 };
};

}

#endif