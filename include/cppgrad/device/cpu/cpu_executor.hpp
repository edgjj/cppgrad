#ifndef CPPGRAD_CPU_EXECUTOR_HPP
#define CPPGRAD_CPU_EXECUTOR_HPP

#include "cppgrad/device/executor.hpp"

namespace cppgrad::impl {

struct CPUExecutor : Executor {

    void copy(std::byte* from, std::byte* to, std::size_t count, CopyType copy_type) override;
    void strided_copy(std::byte* from,
        std::byte* to,
        DType type,
        const std::vector<size_t>& shape,
        const std::vector<size_t>& from_strides,
        const std::vector<size_t>& to_strides) override;

    void fill(std::byte* pos, std::byte* value, DType type, std::size_t count) override;
};

}

#endif