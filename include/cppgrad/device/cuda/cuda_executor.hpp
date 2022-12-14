#ifndef CPPGRAD_CUDA_EXECUTOR_HPP
#define CPPGRAD_CUDA_EXECUTOR_HPP

#include "cppgrad/device/executor.hpp"

namespace cppgrad::impl {

struct CUDAExecutor : Executor {

    void copy(std::byte* from, std::byte* to, std::size_t count, CopyType copy_type) override;
    void strided_copy(const Tensor& from, Tensor& to) override;

    void fill(Tensor& tensor, std::byte* value) override;
};

}

#endif