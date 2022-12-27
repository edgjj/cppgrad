#include "cppgrad/distributed/comm.hpp"
#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/exceptions/mpi_error.hpp"
#include "cppgrad/tensor/tensor.hpp"
#include <iostream>
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

Tensor Communicator::broadcast(const Tensor& tensor, int root_process)
{
    auto cpu_tensor = tensor.cpu();
    cpu_tensor.set_requires_grad(false);

    auto shape_size = cpu_tensor.shape().size();
    uint64_t metadata[3] = { shape_size, cpu_tensor.dtype(), tensor.is_cuda_tensor() };

    CPPGRAD_MPI_CHECK(MPI_Bcast, metadata, 3, MPI_UINT64_T, root_process, _comm);
    // alloc shape/strides
    std::vector<size_t> recv_shape(cpu_tensor.shape()), recv_strides(cpu_tensor._storage->_strides);
    bool is_cuda = metadata[2];

    CPPGRAD_MPI_CHECK(MPI_Bcast, recv_shape.data(), metadata[0], MPI_UINT64_T, root_process, _comm);
    CPPGRAD_MPI_CHECK(MPI_Bcast, recv_strides.data(), metadata[0], MPI_UINT64_T, root_process, _comm);

    // bcast to cpu_tensor
    CPPGRAD_MPI_CHECK(MPI_Bcast, cpu_tensor.data(), cpu_tensor.numel(), dtype_to_mpi((DType)metadata[1]), root_process, _comm);

    cpu_tensor._storage->_strides = recv_strides;

    return is_cuda ? cpu_tensor.cuda() : cpu_tensor;
}

void Communicator::send(const Tensor& tensor, int dest_process, int dest_tag)
{
    // make it cpu; making it contiguous is expensive
    auto cpu_tensor = tensor.cpu();
    cpu_tensor.set_requires_grad(false);

    // we need to send it partially; it's not trivial to send it as a single message
    auto shape_size = cpu_tensor.shape().size();
    uint64_t metadata[3] = { shape_size, cpu_tensor.dtype(), tensor.is_cuda_tensor() };

    CPPGRAD_MPI_CHECK(MPI_Send, metadata, 3, MPI_UINT64_T, dest_process, dest_tag, _comm);

    CPPGRAD_MPI_CHECK(MPI_Send, cpu_tensor.shape().data(), shape_size, MPI_UINT64_T, dest_process, dest_tag, _comm);
    CPPGRAD_MPI_CHECK(MPI_Send, cpu_tensor.strides().data(), shape_size, MPI_UINT64_T, dest_process, dest_tag, _comm);
    CPPGRAD_MPI_CHECK(MPI_Send, cpu_tensor.data(), cpu_tensor.nbytes(), MPI_BYTE, dest_process, dest_tag, _comm);
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

    tensor._storage->_strides = recv_strides;
    return is_cuda ? tensor.cuda() : tensor;
}

Tensor Communicator::gather(const Tensor& tensor, int root_process)
{
    // make it cpu; making it contiguous is expensive
    auto cpu_tensor = tensor.cpu();
    cpu_tensor.set_requires_grad(false);

    auto shape_size = cpu_tensor.shape().size();

    uint64_t metadata_size = rank() == root_process ? size() * 3 : 3;

    // shape_size, dtype, is_cuda
    uint64_t metadata[3] = { shape_size, cpu_tensor.dtype(), tensor.is_cuda_tensor() };
    std::vector<uint64_t> metadata_gather(metadata_size);

    // gather metadata
    CPPGRAD_MPI_CHECK(MPI_Gather, metadata, 3, MPI_UINT64_T, metadata_gather.data(), 3, MPI_UINT64_T, root_process, _comm);

    // this loop 'd run only on root process
    int invalid_meta = 0;
    for (size_t k = 0; k < metadata_gather.size(); k += 3) {
        if (metadata_gather[k] != metadata_gather[0]
            || metadata_gather[k + 1] != metadata_gather[1]
            || metadata_gather[k + 2] != metadata_gather[2]) {
            invalid_meta = true;
            break;
        }
    }

    CPPGRAD_MPI_CHECK(MPI_Bcast, &invalid_meta, 1, MPI_INT32_T, root_process, _comm);

    // stop execution if Tensor is invalid
    if ((bool)invalid_meta) {
        throw exceptions::GenericError(
            rank() == root_process ? "Received Tensor metadata is not equal to other tensors metadata."
                                   : "Attempted to gather mis-shaped Tensor.");
    }

    // need this be non empty on all nodes
    Tensor gather_tensor;

    if (rank() == root_process) {
        std::vector<size_t> gather_shape(tensor.shape());

        gather_shape.insert(gather_shape.begin(), size());
        gather_tensor = Tensor::create_dirty(std::move(gather_shape), (DType)metadata[1], 8, new CPU());

        for (size_t i = 1; i < shape_size + 1; i++) {
            gather_tensor._storage->_strides[i] = tensor._storage->_strides[i - 1];
        }
    } else {
        gather_tensor = Tensor { 0 };
    }

    CPPGRAD_MPI_CHECK(MPI_Gather, cpu_tensor.data(), cpu_tensor.numel(),
        dtype_to_mpi((DType)metadata[1]),
        gather_tensor.data(), cpu_tensor.numel(),
        dtype_to_mpi((DType)metadata[1]),
        root_process, _comm);

    // just return original tensor or discard
    if (rank() != root_process) {
        return tensor;
    }

    bool is_cuda = metadata[2];
    return is_cuda ? gather_tensor.cuda() : gather_tensor;
}

Tensor Communicator::all_gather(const Tensor& tensor)
{
    // make it cpu; making it contiguous is expensive
    auto cpu_tensor = tensor.cpu();
    cpu_tensor.set_requires_grad(false);

    auto shape_size = cpu_tensor.shape().size();

    uint64_t metadata_size = size() * 3;

    // shape_size, dtype, is_cuda
    uint64_t metadata[3] = { shape_size, cpu_tensor.dtype(), tensor.is_cuda_tensor() };
    std::vector<uint64_t> metadata_gather(metadata_size);

    // gather metadata
    CPPGRAD_MPI_CHECK(MPI_Allgather, metadata, 3, MPI_UINT64_T, metadata_gather.data(), 3, MPI_UINT64_T, _comm);

    // this loop 'd run only on root process
    int invalid_meta = 0;
    for (size_t k = 0; k < metadata_gather.size(); k += 3) {
        if (metadata_gather[k] != metadata_gather[0]
            || metadata_gather[k + 1] != metadata_gather[1]
            || metadata_gather[k + 2] != metadata_gather[2]) {
            throw exceptions::GenericError("Received Tensor metadata is not equal to other tensors metadata.");
            break;
        }
    }

    std::vector<size_t> gather_shape(cpu_tensor.shape());
    gather_shape.insert(gather_shape.begin(), size());

    // need this be non empty on all nodes
    Tensor gather_tensor = Tensor::create_dirty(std::move(gather_shape), (DType)metadata[1], 8, new CPU());
    for (size_t i = 1; i < shape_size + 1; i++) {
        gather_tensor._storage->_strides[i] = tensor._storage->_strides[i - 1];
    }

    CPPGRAD_MPI_CHECK(MPI_Allgather, cpu_tensor.data(), cpu_tensor.numel(),
        dtype_to_mpi((DType)metadata[1]),
        gather_tensor.data(), cpu_tensor.numel(),
        dtype_to_mpi((DType)metadata[1]),
        _comm);

    bool is_cuda = metadata[2];
    return is_cuda ? gather_tensor.cuda() : gather_tensor;
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