#ifndef CPPGRAD_TENSOR_STRIDED_SPAN_HPP
#define CPPGRAD_TENSOR_STRIDED_SPAN_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace cppgrad {

namespace impl {

    template <typename Type, typename = void>
    struct IsTensor : std::false_type {
    };

    template <typename Type>
    struct IsTensor<Type,
        typename std::enable_if_t<std::is_member_function_pointer_v<decltype(&Type::numel)>>>
        : std::true_type {
    };

    template <typename T>
    inline constexpr bool is_tensor_v = IsTensor<T>::value;

}

template <typename T>
struct StridedSpan {

    StridedSpan(const StridedSpan<T>&) = default;
    StridedSpan(StridedSpan<T>&&) = default;

    template <typename Tensor,
        std::enable_if_t<impl::is_tensor_v<Tensor>>* = 0>
    StridedSpan(Tensor& t)
        : _data(reinterpret_cast<T*>(t.data()))
        , _size(t.numel())
    {
        _stride = std::is_sorted(t.strides().begin(), t.strides().end()) ? *t.strides().rbegin() : *t.strides().begin();
        _stride /= sizeof(T);
    }

    T& operator[](size_t index)
    {
        return *(_data + index * _stride);
    }

    size_t size() const
    {
        return _size;
    }

    bool is_contiguous() const
    {
        return _stride == 1;
    }

    T* data()
    {
        return _data;
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

    template <typename Tensor,
        std::enable_if_t<impl::is_tensor_v<Tensor>>* = 0>
    StridedSpan2D(Tensor& t)
        : _data(reinterpret_cast<T*>(t.data()))
        , _strides { t.strides()[0] / sizeof(T), t.strides()[1] / sizeof(T) }
        , _sizes { t.shape()[0], t.shape()[1] }
    {
        // check if tensor is 2-dim; no exception due to possible poor perf
        assert(t.strides().size() == 2);
    }

    T& operator()(size_t row, size_t col)
    {
        return *(_data + row * _strides[0] + col * _strides[1]);
    }

    bool is_contiguous() const
    {
        return _strides[1] == 1;
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