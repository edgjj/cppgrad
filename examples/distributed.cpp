#ifdef CPPGRAD_HAS_MPI

#include <cppgrad/cppgrad.hpp>
#include <iomanip>
#include <iostream>

using namespace cppgrad;

template <DType DataType>
void print_tensor(const Tensor& data)
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

int main(int argc, char* argv[])
{
    try {
        // init mpi context
        distributed::Environment env(argc, argv);
        distributed::Communicator world;

        auto tg = Tensor::create<f32>({ 2, 2 });
        tg.random_fill();

        auto gathered = world.gather(tg, 0);
        if (world.rank() == 0) {
            std::cout << "Gathered Tensor: " << std::endl;
            print_tensor<f32>(gathered);
        }

        if (world.size() < 2) {
            std::cout << "Not enough processes to work, needed at least 2.";
            return -1;
        }

        if (world.rank() == 0) {
            auto t0 = Tensor::create<f32>({ 8, 8 }, 98);
#ifdef CPPGRAD_HAS_CUDA
            t0 = t0.cuda(); // test CUDA tensor send
#endif
            t0.random_fill();
            world.send(t0, 1);

            std::cout << "Sent Tensor from process 0 to process 1" << std::endl;
        } else if (world.rank() == 1) {
            auto t1 = world.recv(0);
            std::cout << "Received Tensor from process 0; DType: " << dtype_name(t1.dtype()) << "; device type: " << t1.device().type() << std::endl;

            print_tensor<f32>(t1);
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