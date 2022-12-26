#ifdef CPPGRAD_HAS_MPI

#include <cppgrad/cppgrad.hpp>
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
        distributed::Environment(argc, argv);
        distributed::Communicator world;

        if (world.rank() == 0) {
            std::cout << "I'm 0!" << std::endl;
        } else {
            std::cout << "I'm " << world.rank() << std::endl;
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