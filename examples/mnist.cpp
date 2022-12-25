#include <cppgrad/cppgrad.hpp>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "deps/stb_image.h"

using namespace cppgrad;

template <DType DataType>
void print_tensor(Tensor& data)
{
    std::cout << std::setprecision(8);
    if (data.shape().size() > 1) {
        for (size_t k = 0; k < data.shape()[0]; k++) {
            print_tensor<DataType>(data[k]);
        }
    } else {
        std::cout << "[ ";
        for (size_t k = 0; k < data.numel(); k++) {
            std::cout << (double)data[k].item<DataType>() << ' ';
        }
        std::cout << " ]" << std::endl;
    }
}

#define MNIST_W 28
#define MNIST_H 28

struct LinearNN : nn::Module {
    LinearNN()
        : fc1(MNIST_W * MNIST_H, 256)
        , fc2(256, 64)
        , fc3(64, 10)
    {
        register_module(fc1);
    }

    nn::tensor_list forward(nn::tensor_list inputs) override
    {
        auto y = fc1(inputs);
        y = fc2(cppgrad::relu(y[0]));
        y = fc3(cppgrad::relu(y[0]));
        return { cppgrad::tanh(y[0]) };
    }

private:
    nn::Linear fc1;
    nn::Linear fc2;
    nn::Linear fc3;
};

int main()
{
    try {
        autograd::ForceGradGuard guard;

        auto t1 = Tensor::create<f32>({ 1, MNIST_W * MNIST_H }, 1.0f);
        t1.random_fill();

        auto y = Tensor{ { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } };

        LinearNN nn;
        nn::optim::SGD optim(nn, 1e-2);
        // nn.cuda() switches all params to CUDA

        constexpr size_t n_steps = 150;
        for (size_t k = 0; k < n_steps; k++) {

            auto output = nn(t1)[0];
            auto loss = nn::mse_loss(output, y);

            optim.zero_grad();

            loss.backward();
            optim.step();

            std::cout << "Output: ";
            print_tensor<f32>(output);

            std::cout << "Loss: ";
            print_tensor<f32>(loss);

            std::cout << "Real Y: "; 
            print_tensor<f32>(y);
        }

        return 0;
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}