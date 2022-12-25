#include <cppgrad/cppgrad.hpp>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace cppgrad;

struct LinearNN : nn::Module {
    LinearNN()
        : fc1(30, 1)
    {
        register_module(fc1);
    }

    nn::tensor_list forward(nn::tensor_list inputs) override
    {
        auto& x = inputs[0];

        auto y = fc1(std::move(inputs))[0];
        y = cppgrad::relu(y);

        return { y };
    }

private:
    nn::Linear fc1;
};

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

int main()
{
    try {
        autograd::ForceGradGuard guard;

        auto t1 = Tensor::create<f32>({ 1, 30 }, 23.0f);
        auto y = Tensor({ { 23.0f } });

        LinearNN nn;
        nn::optim::SGD optim(nn, 1e-3);
        // nn.cuda() switches all params to CUDA

        constexpr size_t n_steps = 150;
        for (size_t k = 0; k < n_steps; k++) {
            auto output = nn({ t1 })[0];
            output = cppgrad::relu(output);

            auto loss = nn::mse_loss(output, y);

            optim.zero_grad();
            loss.backward();
            optim.step();

            std::cout << "Output: " << output.item<f32>() << std::endl;
            std::cout << "Loss: " << loss.item<f32>() << std::endl;

            std::cout << "Real Y: " << y.item<f32>() << std::endl;
        }

        return 0;
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}