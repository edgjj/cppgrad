// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_CUDA_EXECUTOR_HPP
#define CPPGRAD_CUDA_EXECUTOR_HPP

#include "cppgrad/device/executor.hpp"

namespace cppgrad {
struct CUDA;
}

namespace cppgrad::impl {

struct CUDAExecutor : Executor {

    CUDAExecutor(CUDA& parent_device);

    void copy(const std::byte* from, std::byte* to, std::size_t count, CopyType copy_type) override;
    void strided_copy(const Tensor& from, Tensor& to) override;

    void fill(Tensor& tensor, std::byte* value) override;
    void random_fill(Tensor& tensor, double lower_bound, double upper_bound) override;

    void add(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void sub(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void mul(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void div(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;

    void pow(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;

    void dot(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void sum(const Tensor& lhs, Tensor& dst) override;

    void log(const Tensor& lhs, Tensor& dst) override;
    void exp(const Tensor& lhs, Tensor& dst) override;

    void relu(const Tensor& lhs, Tensor& dst) override;
    void tanh(const Tensor& lhs, Tensor& dst) override;
    void sign(const Tensor& lhs, Tensor& dst) override;
    void neg(const Tensor& lhs, Tensor& dst) override;

    void cmp(const Tensor& lhs, const Tensor& rhs, Tensor& dst, CompareType cmp_type) override;

private:
    CUDA& _parent;
};

}

#endif