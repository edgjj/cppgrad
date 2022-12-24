#ifndef CPPGRAD_CPU_EXECUTOR_HPP
#define CPPGRAD_CPU_EXECUTOR_HPP

#include "cppgrad/device/executor.hpp"

namespace cppgrad::impl {

/// @brief for reference it's naive by default,
/// to be splitted into multiple CPU (SSE4/AVX/AVX2) backends
struct CPUExecutor : Executor {

    void copy(const std::byte* from, std::byte* to, std::size_t count, CopyType copy_type) override;
    void strided_copy(const Tensor& from, Tensor& to) override;

    void fill(Tensor& tensor, std::byte* value) override;

    void sum(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void sub(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void mul(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void div(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;

    void pow(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void dot(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;
    void matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst) override;

    void log(const Tensor& lhs, Tensor& dst) override;
    void exp(const Tensor& lhs, Tensor& dst) override;

    void relu(const Tensor& lhs, Tensor& dst) override;
    void tanh(const Tensor& lhs, Tensor& dst) override;
    void cmp(const Tensor& lhs, const Tensor& rhs, Tensor& dst, CompareType cmp_type) override;
};

}

#endif