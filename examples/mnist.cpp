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

int main()
{
    try {
        autograd::ForceGradGuard guard;

        auto x = Tensor::create<f32>({ 1, MNIST_W * MNIST_H }, 0.5f);
        auto y = Tensor { { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f } };

        LinearNN nn;
        nn::optim::SGD optim(nn, 5e-3);

        // nn.cuda() switches all params to CUDA
#ifdef CPPGRAD_HAS_CUDA
        x = x.cuda();
        y = y.cuda();
        nn.cuda();
#endif

        constexpr size_t n_steps = 50000;
        constexpr size_t print_threshold = 250;

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();

        for (size_t k = 0; k < n_steps; k++) {

            auto output = nn(x)[0];
            auto loss = nn::mse_loss(output, y);

            optim.zero_grad();

            loss.backward();
            optim.step();

            if (k != 0 && k % print_threshold == 0) {
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