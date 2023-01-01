// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_OP_WRAPPER_HPP
#define CPPGRAD_OP_WRAPPER_HPP

#include "cppgrad/tensor/util/strided_span.hpp"
#include <utility>

namespace cppgrad {

/**
 * @brief 2-input 1-output operation wrapper.
 *
 * @tparam Fn target function
 * @tparam Tensor
 */
template <typename Fn, typename Tensor>
struct OpWrapperBase {

    /**
     * @brief Constructs a new wrapper object
     *
     * @param fun target function
     * @param out output Tensor
     * @param lhs left input Tensor
     * @param rhs right input Tensor
     */
    OpWrapperBase(Fn&& fun, Tensor& out, const Tensor& lhs, const Tensor& rhs)
        : _fun(std::forward<Fn>(fun))
        , _out(out)
        , _lhs(lhs)
        , _rhs(rhs)
    {
    }

protected:
    Fn _fun;

    Tensor& _out;
    const Tensor& _lhs;
    const Tensor& _rhs;
};

template <typename Fn, typename Tensor>
struct OpWrapper1D : OpWrapperBase<Fn, Tensor> {
    using OpWrapperBase<Fn, Tensor>::OpWrapperBase;

    template <typename T>
    void operator()(T tag)
    {
        ConstStridedSpan<T> op1 { this->_lhs };
        ConstStridedSpan<T> op2 { this->_rhs };

        StridedSpan<T> out { this->_out };

        this->_fun(out, op1, op2);
    }
};

template <typename Fn, typename Tensor>
OpWrapper1D(Fn&&, Tensor&, const Tensor&, const Tensor&) -> OpWrapper1D<Fn, Tensor>;

template <typename Fn, typename Tensor>
struct OpWrapper2D : OpWrapperBase<Fn, Tensor> {
    using OpWrapperBase<Fn, Tensor>::OpWrapperBase;

    template <typename T>
    void operator()(T tag)
    {
        ConstStridedSpan2D<T> op1 { this->_lhs };
        ConstStridedSpan2D<T> op2 { this->_rhs };

        StridedSpan2D<T> out { this->_out };

        this->_fun(out, op1, op2);
    }
};

template <typename Fn, typename Tensor>
OpWrapper2D(Fn&&, Tensor&, const Tensor&, const Tensor&) -> OpWrapper2D<Fn, Tensor>;

}

#endif