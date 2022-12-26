#include "cppgrad/distributed/comm.hpp"
#include "cppgrad/exceptions/mpi_error.hpp"
#include "cppgrad/tensor/tensor.hpp"
#include <mpi.h>

namespace cppgrad::distributed {

static MPI_Datatype dtype_to_mpi(DType dtype)
{
    switch (dtype) {
    case DType::f32:
        return MPI_FLOAT;
    case DType::f64:
        return MPI_DOUBLE;
    case DType::i32:
        return MPI_INT32_T;
    case DType::i64:
        return MPI_INT64_T;
    case DType::u32:
        return MPI_UINT32_T;
    case DType::u64:
        return MPI_UINT64_T;
    }

    return MPI_BYTE;
}

Communicator::Communicator()
{
    _comm = MPI_COMM_WORLD;
}

Communicator::~Communicator()
{
    if (_comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&_comm);
    }
}

Tensor Communicator::broadcast(const Tensor& tensor)
{
    return Tensor();
}

void Communicator::send(const Tensor& tensor, int dest_process, int dest_tag)
{
    // make it cpu; making it contiguous is expensive
    auto cpu_tensor = tensor.cpu();

    // we need to send it partially; it's not trivial to send it as a single message
    auto shape_size = cpu_tensor.shape().size();
    uint64_t metadata[3] = { shape_size, cpu_tensor.dtype(), cpu_tensor.is_cuda_tensor() };

    CPPGRAD_MPI_CHECK(MPI_Send, metadata, 3, MPI_UINT64_T, dest_process, dest_tag, _comm);

    CPPGRAD_MPI_CHECK(MPI_Send, cpu_tensor.shape().data(), shape_size, MPI_UINT64_T, dest_process, dest_tag, _comm);
    CPPGRAD_MPI_CHECK(MPI_Send, cpu_tensor.strides().data(), shape_size, MPI_UINT64_T, dest_process, dest_tag, _comm);
    CPPGRAD_MPI_CHECK(MPI_Send, nullptr, cpu_tensor.nbytes(), MPI_BYTE, dest_process, dest_tag, _comm);
}

Tensor Communicator::recv(int source_process, int source_tag)
{
    MPI_Status status;

    // shape_size, dtype, is_cuda
    uint64_t metadata[3] = { 0, 0, 0 };

    CPPGRAD_MPI_CHECK(MPI_Recv, metadata, 3, MPI_UINT64_T, source_process, source_tag, _comm, &status);
    bool is_cuda = metadata[2];

    // alloc shape/strides
    std::vector<size_t> recv_shape(metadata[0]), recv_strides(metadata[0]);

    // recv shape/strides
    CPPGRAD_MPI_CHECK(MPI_Recv, recv_shape.data(), metadata[0], MPI_UINT64_T, source_process, source_tag, _comm, &status);
    CPPGRAD_MPI_CHECK(MPI_Recv, recv_strides.data(), metadata[0], MPI_UINT64_T, source_process, source_tag, _comm, &status);

    auto tensor = Tensor::create_dirty(std::move(recv_shape), (DType)metadata[1], 8, new CPU());
    CPPGRAD_MPI_CHECK(MPI_Recv, tensor.data(), tensor.numel(), dtype_to_mpi((DType)metadata[1]), source_process, source_tag, _comm, &status);

    return is_cuda ? tensor.cuda() : tensor;
}

Tensor Communicator::gather(const Tensor& tensor)
{
    return Tensor();
}

Tensor Communicator::all_gather(const Tensor& tensor)
{
    return Tensor();
}

int Communicator::size() const
{
    int ret = 0;
    CPPGRAD_MPI_CHECK(MPI_Comm_size, _comm, &ret);

    return ret;
}

int Communicator::rank() const
{
    int ret = 0;
    CPPGRAD_MPI_CHECK(MPI_Comm_rank, _comm, &ret);

    return ret;
}

}