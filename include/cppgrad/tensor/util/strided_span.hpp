#ifndef CPPGRAD_TENSOR_STRIDED_SPAN_HPP
#define CPPGRAD_TENSOR_STRIDED_SPAN_HPP

#include "cppgrad/tensor/tensor_fwd.hpp"
#include <cstddef>
#include <type_traits>
#include <vector>

namespace cppgrad {

template <typename T>
struct StridedSpan {

    using BytePtr = std::conditional_t<std::is_const_v<T>, const std::byte*, std::byte*>;

    StridedSpan(BytePtr data, const std::vector<size_t>& strides, const std::vector<size_t>& sizes)
        : _data(reinterpret_cast<T*>(data))
        , _stride(strides[0] / sizeof(T))
        , _size(sizes[0])
    {
    }

    T& operator[](size_t index)
    {
        return *(_data + index * _stride);
    }

    size_t size() const
    {
        return _size;
    }

private:
    T* _data;
    size_t _stride;
    size_t _size;
};

// bad thing that we got to have both of these

template <typename T>
struct ConstStridedSpan : StridedSpan<const T> {
    using StridedSpan::StridedSpan;
};

template <typename T>
struct StridedSpan2D {

    using BytePtr = std::conditional_t<std::is_const_v<T>, const std::byte*, std::byte*>;

    StridedSpan2D(BytePtr data, const std::vector<size_t>& strides, const std::vector<size_t>& sizes)
        : _data(reinterpret_cast<T*>(data))
        , _strides{ strides[0] / sizeof (T), strides[1] / sizeof(T) }
        , _sizes{ sizes[0], sizes[1] }
    {
    }

    T& operator()(size_t row, size_t col)
    {
        return *(_data + row * _strides[0] + col * _strides[1]);
    }

    size_t size(size_t dim)
    {
        return _sizes[dim];
    }

private:
    T* _data;
    size_t _strides[2];
    size_t _sizes[2];
};

template <typename T>
struct ConstStridedSpan2D : StridedSpan2D<const T> {
    using StridedSpan2D::StridedSpan2D;
};

}

#endif