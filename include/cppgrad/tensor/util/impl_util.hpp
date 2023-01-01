// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_TENSOR_IMPL_UTIL_HPP
#define CPPGRAD_TENSOR_IMPL_UTIL_HPP

#include "cppgrad/tensor/typing.hpp"
#include <vector>

namespace cppgrad {

/**
 * @brief Internal macro to check equality
 *
 */
#define CPPGRAD_CHECK_EQ(lhs, rhs, exception_type, ...) \
    if (lhs != rhs)                                     \
        throw exception_type(__VA_ARGS__);

#define CPPGRAD_CHECK_FALSE(val, exception_type, ...) \
    if (val)                                          \
        throw exception_type(__VA_ARGS__);

namespace impl {

    /**
     * @brief Internal function to make Tensor strides.
     *
     * @param shape Requested tensor shape
     * @param type_size Requested tensor type size in bytes
     * @return std::vector<size_t> Strides
     */
    static std::vector<size_t> make_strides(std::vector<size_t> shape, size_t type_size)
    {
        std::vector<size_t> strides(shape.size());
        size_t accum = type_size;

        auto it_stride = strides.rbegin();

        for (auto it = shape.rbegin(); it != shape.rend(); it++) {
            *it_stride = accum;
            accum *= *it;
            it_stride++;
        }

        return strides;
    }

}

}

#endif