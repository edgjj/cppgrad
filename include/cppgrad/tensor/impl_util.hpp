#ifndef CPPGRAD_TENSOR_IMPL_UTIL_HPP
#define CPPGRAD_TENSOR_IMPL_UTIL_HPP

#include "cppgrad/tensor/typing.hpp"
#include <vector>

namespace cppgrad::impl {

/**
 * @brief Internal function to make Tensor strides.
 *
 * @param shape Requested tensor shape
 * @param type_size Requested tensor type size in bytes
 * @return std::vector<size_t> Strides
 */
std::vector<size_t> make_strides(std::vector<size_t> shape, size_t type_size)
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

#endif