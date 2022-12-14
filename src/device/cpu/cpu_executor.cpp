#include "cppgrad/device/cpu/cpu_executor.hpp"
#include "cppgrad/tensor/tensor.hpp"

#include <algorithm>
#include <cstring>

namespace cppgrad::impl {

template <typename T>
static void fill_impl(std::byte* pos, std::byte* value, std::size_t count)
{
    auto* ptr = reinterpret_cast<T*>(pos);
    auto fill_value = *reinterpret_cast<T*>(value);

    // std::fill_n(OutputIt, Size, T& value)
    std::fill_n(ptr, count, fill_value);
}

template <typename T>
static void strided_copy_copier(T* from,
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
static void strided_copy_impl(std::byte* from,
    std::byte* to,
    const size_t* shape,
    const size_t* from_strides,
    const size_t* to_strides,
    size_t shape_size)
{
    if (shape_size == 1) {
        strided_copy_copier<T>(reinterpret_cast<T*>(from),
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

void CPUExecutor::copy(std::byte* from, std::byte* to, std::size_t count, CopyType copy_type)
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
    FOREACH_TYPE(tensor.dtype(), impl::fill_impl, tensor.data(), value, tensor.numel());
}

}