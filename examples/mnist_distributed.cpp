#ifdef CPPGRAD_HAS_MPI

#include <cppgrad/cppgrad.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "deps/stb_image.h"

using namespace cppgrad;

template <DType DataType>
void print_tensor(const Tensor& data)
{
    std::cout << std::setprecision(4);
    if (data.shape().size() > 1) {
        for (size_t k = 0; k < data.shape()[0]; k++) {
            print_tensor<DataType>(data[k]);
        }
    } else {
        std::cout << "[ ";
        size_t max_size = 10;
        size_t size = data.numel() > max_size ? max_size : data.numel();

        for (size_t k = 0; k < size; k++) {
            std::cout << (double)data[k].item<DataType>() << ' ';
        }

        if (data.numel() > max_size) {
            std::cout << " ... ";
        }

        std::cout << " ]" << std::endl;
    }
}

template <typename Net>
struct DistributedSGD {
    DistributedSGD(Net& net, distributed::Communicator& comm, double learning_rate)
        : _lr(learning_rate)
        , _comm(comm)
    {
        _params = net.get_parameters();
    }

    void step()
    {
        for (auto& i : _params) {
            auto lr_tensor = Tensor::create_dirty(i->grad().shape(), i->dtype(), 8, i->device().clone());
            lr_tensor.fill(_lr);

            auto calculated_change = i->grad() * lr_tensor;
            auto gathered_change = _comm.gather(calculated_change, 0);

            if (_comm.rank() == 0) {
                // iterate over gathered list
                for (size_t k = 0; k < _comm.size(); k++) {
                    (*i) -= gathered_change[k];
                }
            }

            auto bcasted = *i;
            (*i) = Tensor();
            (*i) = _comm.broadcast(bcasted, 0);

            // restore state
            i->set_requires_grad(true);
        }
    }

    void zero_grad()
    {
        // sets grads to nothing
        for (auto& i : _params) {
            (*i) = (*i);
        }
    }

private:
    distributed::Communicator& _comm;

    nn::tensor_ptr_list _params;
    double _lr { 0.0 };
};

#define MNIST_W 28
#define MNIST_H 28

struct LinearNN : nn::Module {
    LinearNN()
        : fc1(MNIST_W * MNIST_H, 256)
        , fc2(256, 64)
        , fc3(64, 10)
    {
        // module reg is important !
        register_module(fc1);
        register_module(fc2);
        register_module(fc3);
    }

    nn::tensor_list forward(nn::tensor_list inputs) override
    {
        auto y = fc1(inputs);
        y = fc2(cppgrad::relu(y[0]));
        y = fc3(cppgrad::relu(y[0]));
        return { cppgrad::sigmoid(y[0]) };
    }

    // private:
    nn::Linear fc1;
    nn::Linear fc2;
    nn::Linear fc3;
};

int main(int argc, char* argv[])
{
    try {
        distributed::Environment env(argc, argv);
        distributed::Communicator world;

        auto x = Tensor::create<f32>({ 1, MNIST_W * MNIST_H }, 0.5f);
        auto y = Tensor { { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f } };

        LinearNN nn;
        DistributedSGD optim(nn, world, 1e-4);

        // nn.cuda() switches all params to CUDA
#ifdef CPPGRAD_HAS_CUDA
        x = x.cuda();
        y = y.cuda();
        nn.cuda();
#endif

        constexpr size_t n_steps = 50000;
        constexpr size_t print_threshold = 125;

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        for (size_t k = 0; k < n_steps; k++) {

            auto output = nn(x)[0];
            auto loss = nn::mse_loss(output, y);

            optim.zero_grad();

            loss.backward();
            optim.step();

            if (k != 0 && k % print_threshold == 0 && world.rank() == 0) {
                std::cout << "Metrics @ " << world.rank() << std::endl;

                end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> run_ms = end - start;
                start = end;

                std::cout << "Time per " << print_threshold << " steps: " << run_ms.count() << "ms" << std::endl;

                std::cout << "Output: ";
                print_tensor<f32>(output);

                std::cout << "Real Y: ";
                print_tensor<f32>(y);

                std::cout << "Loss: ";
                print_tensor<f32>(loss);

                std::cout << std::endl;
            }
        }

        return 0;
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}

#endif