// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_DISTRIBUTED_COMM_HPP
#define CPPGRAD_DISTRIBUTED_COMM_HPP

#ifdef CPPGRAD_HAS_MPI

#include "cppgrad/tensor/tensor_fwd.hpp"

namespace cppgrad::distributed {

using CommType = int;

/**
 * @brief MPI Communicator wrapper, allows distributed computing with Tensors.
 *
 */
struct Communicator {
    Communicator();
    ~Communicator();

    /**
     * @brief Broadcasts Tensor along communicator
     *
     * Note: blocking routine.
     * @param tensor
     * @return Tensor
     */
    Tensor broadcast(const Tensor& tensor, int root_process);

    /**
     * @brief Sends Tensor to given process & tag in communicator
     *
     * Note: blocking routine.
     * @param tensor
     */
    void send(const Tensor& tensor, int dest_process, int dest_tag = 0);

    /**
     * @brief Receives Tensor from given process & tag in communicator
     *
     * Note: blocking routine.
     * @param tensor
     * @return Tensor
     */
    Tensor recv(int src_process, int src_tag = 0);

    /**
     * @brief Gathers Tensor to root communicator process;
     * This joins multiple processes Tensors into single one on root processor
     *
     * @param tensor
     * @return Tensor
     */
    Tensor gather(const Tensor& tensor, int root_process);

    /**
     * @brief Gathers Tensor along processes in communicator
     *
     * @param tensor
     * @return Tensor
     */
    Tensor all_gather(const Tensor& tensor);

    /**
     * @brief Determines the size of communicator process group
     *
     * @return int
     */
    int size() const;

    /**
     * @brief Tells process rank within communicator
     *
     * @return int
     */
    int rank() const;

private:
    CommType _comm;
};

}

#endif

#endif