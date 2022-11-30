#ifndef CPPGRAD_TENSOR_TYPES_HPP
#define CPPGRAD_TENSOR_TYPES_HPP

#include <cstdint>

namespace cppgrad::impl {

using cppgrad_id = uint32_t;

// will probably cause weird error message on invalid types but we're with that
template <typename>
struct Type {
};

/*
    let it be 32/64 bit unsigned/signed int & 32/64 bit floats
*/
template <>
struct Type<uint32_t> {
    static constexpr cppgrad_id id = 0;
};

template <>
struct Type<int32_t> {
    static constexpr cppgrad_id id = 1;
};

template <>
struct Type<uint64_t> {
    static constexpr cppgrad_id id = 5;
};

template <>
struct Type<int64_t> {
    static constexpr cppgrad_id id = 6;
};

template <>
struct Type<float> {
    static constexpr cppgrad_id id = 10;
};

template <>
struct Type<double> {
    static constexpr cppgrad_id id = 11;
};

}

#endif