// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_ITERTOOLS_REVERSED_HPP
#define CPPGRAD_ITERTOOLS_REVERSED_HPP

namespace cppgrad::itertools {

template <typename T>
struct reversion_wrapper {
    T&& iterable;
};

template <typename T>
auto begin(reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

template <typename T>
auto end(reversion_wrapper<T> w) { return std::rend(w.iterable); }

template <typename T>
reversion_wrapper<T> reversed(T&& iterable) { return { std::forward<T>(iterable) }; }

}

#endif