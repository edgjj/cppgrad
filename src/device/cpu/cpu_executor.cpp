#include "cppgrad/device/cpu/cpu_executor.hpp"
#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/tensor/ops/op_wrapper.hpp"
#include "cppgrad/tensor/tensor.hpp"
#include "cppgrad/tensor/util/strided_span.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>

namespace cppgrad::impl {

void CPUExecutor::copy(const std::byte* from, std::byte* to, std::size_t count, CopyType copy_type)
{
    // ignore copy_type as it's CPU
    std::memcpy(to, from, count);
}

void CPUExecutor::strided_copy(const Tensor& from, Tensor& to)
{
    auto fn = [](auto out, auto p1, auto p2) {
        for (size_t k = 0; k < out.size(); k++) {
            out[k] = p1[k];
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), to, from, from }, from.dtype());
}

void CPUExecutor::fill(Tensor& tensor, std::byte* value)
{
    auto fn = [value](auto out, auto p1, auto p2) {
        using Type = typename decltype(out)::Type;
        auto fill_value = *reinterpret_cast<Type*>(value);

        for (size_t k = 0; k < out.size(); k++) {
            out[k] = fill_value;
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), tensor, tensor, tensor }, tensor.dtype());
}

void CPUExecutor::sum(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        for (size_t k = 0; k < out.size(); k++) {
            out[k] = p1[k] + p2[k];
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CPUExecutor::sub(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        for (size_t k = 0; k < out.size(); k++) {
            out[k] = p1[k] - p2[k];
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CPUExecutor::mul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        for (size_t k = 0; k < out.size(); k++) {
            out[k] = p1[k] * p2[k];
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CPUExecutor::pow(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        for (size_t k = 0; k < out.size(); k++) {
            out[k] = std::pow(p1[k], p2[k]);
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CPUExecutor::dot(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        using Type = typename decltype(out)::Type;

        out[0] = Type(0);
        for (size_t k = 0; k < p1.size(); k++) {
            out[0] += p1[k] * p2[k];
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CPUExecutor::matmul(const Tensor& lhs, const Tensor& rhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        using Type = typename decltype(out)::Type;

        for (size_t i = 0; i < out.size(0); i++) { // row
            for (size_t j = 0; j < out.size(1); j++) { // col
                // null elem
                out(i, j) = Type(0);

                for (size_t k = 0; k < p2.size(0); k++) { // row-col advance
                    out(i, j) += p1(i, k) * p2(k, j);
                }
            }
        }
    };

    for_each_type(OpWrapper2D { std::move(fn), dst, lhs, rhs }, dst.dtype());
}

void CPUExecutor::relu(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        using Type = typename decltype(out)::Type;

        for (size_t k = 0; k < out.size(); k++) {
            out[k] = std::max(Type(0), p1[k]);
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CPUExecutor::tanh(const Tensor& lhs, Tensor& dst)
{
    auto fn = [](auto out, auto p1, auto p2) {
        for (size_t k = 0; k < out.size(); k++) {
            out[k] = std::tanh(p1[k]);
        }
    };

    for_each_type(OpWrapper1D { std::move(fn), dst, lhs, lhs }, dst.dtype());
}

void CPUExecutor::cmp(const Tensor& lhs, const Tensor& rhs, Tensor& dst, CompareType cmp_type)
{
}

}