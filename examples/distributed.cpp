// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifdef CPPGRAD_HAS_MPI

#include <cppgrad/cppgrad.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace cppgrad;

template <DType DataType>
void tensor_to_string(const Tensor& data, std::stringstream& ss, bool is_last = true)
{
    if (data.shape().size() > 1) {
        for (size_t k = 0; k < data.shape()[0]; k++) {
            tensor_to_string<DataType>(data[k], ss, k == data.shape()[0] - 1);
        }
    } else {
        ss << std::setw(6);
        ss << "[ ";
        for (size_t k = 0; k < data.numel(); k++) {
            ss << (double)data[k].item<DataType>() << ' ';
        }
        ss << " ]";
        if (!is_last) {
            ss << ",";
        }

        ss << std::endl;
    }
}

template <DType DataType>
std::string tensor_to_string(const Tensor& data)
{
    std::stringstream ss;
    ss << std::setprecision(8);
    ss << "[ " << std::endl;

    tensor_to_string<DataType>(data, ss);
    ss << " ]";
    return ss.str();
}

int main(int argc, char* argv[])
{
    try {
        // init mpi context
        distributed::Environment env(argc, argv);
        distributed::Communicator world;

        auto tg = Tensor::full({ 2, 2 }, 0, f32);
        tg.random_fill();

        auto gathered = world.gather(tg, 0);
        if (world.rank() == 0) {
            std::cout << "Gathered Tensor: \n"
                      << tensor_to_string<f32>(gathered) << std::endl;
        }

        auto all_gathered = world.all_gather(tg);
        std::cout << "All-Gathered Tensor @ rank " << world.rank() << ": \n"
                  << tensor_to_string<f32>(all_gathered) << std::endl;

        if (world.size() < 2) {
            std::cout << "Not enough processes to work, needed at least 2.";
            return -1;
        }

        if (world.rank() == 0) {
            auto t0 = Tensor::full({ 8, 8 }, 98, f32);
#ifdef CPPGRAD_HAS_CUDA
            t0 = t0.cuda(); // test CUDA tensor send
#endif
            t0.random_fill();
            world.send(t0, 1);

            std::cout << "Sent Tensor from process 0 to process 1" << std::endl;
        } else if (world.rank() == 1) {
            auto t1 = world.recv(0);
            std::cout << "Received Tensor from process 0; DType: " << dtype_name(t1.dtype()) << "; device type: " << t1.device() << std::endl;

            std::cout << tensor_to_string<f32>(t1) << std::endl;
        }
    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}
#else

// build empty
int main()
{
}

#endif