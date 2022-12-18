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

template <typename T>
static void strided_copy_copier(const T* from,
    T* to,
    size_t count,
    size_t from_stride,
    size_t to_stride)
{
    for (size_t i = 0; i < count; i++) {
        to[i * to_stride] = from[i * from_stride];
    }
}

template <typename T>
static void strided_copy_impl(const std::byte* from,
    std::byte* to,
    const size_t* shape,
    const size_t* from_strides,
    const size_t* to_strides,
    size_t shape_size)
{
    if (shape_size == 1) {
        strided_copy_copier<T>(reinterpret_cast<const T*>(from),
            reinterpret_cast<T*>(to),
            *shape,
            *from_strides / sizeof(T),
            *to_strides / sizeof(T));

        return;
    }

    while (shape_size != 1) {
        strided_copy_impl<T>(from + *from_strides, to + *to_strides,
            ++shape,
            ++from_strides,
            ++to_strides,
            --shape_size);
    }
}

void CPUExecutor::copy(const std::byte* from, std::byte* to, std::size_t count, CopyType copy_type)
{
    // ignore copy_type as it's CPU
    std::memcpy(to, from, count);
}

void CPUExecutor::strided_copy(const Tensor& from, Tensor& to)
{
    FOREACH_TYPE(from.dtype(),
        impl::strided_copy_impl,
        from.data(),
        to.data(),
        from.shape().data(),
        from.strides().data(),
        to.strides().data(),
        from.shape().size());
}

void CPUExecutor::fill(Tensor& tensor, std::byte* value)
{
    auto fn = [&](auto tag) {
        using Type = decltype(tag);

        auto* ptr = reinterpret_cast<Type*>(tensor.data());
        auto fill_value = *reinterpret_cast<Type*>(value);

        // std::fill_n(OutputIt, Size, T& value)
        std::fill_n(ptr, tensor.numel(), fill_value);
    };

    for_each_type(std::move(fn), tensor.dtype());
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
    auto fn = [&](auto out, auto p1, auto p2) {
        using Type = typename decltype(out)::Type;

        for (size_t i = 0; i < p1.size(0); i++) { // row
            for (size_t j = 0; j < p2.size(1); j++) { // col
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