// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_VALUE_IMPL_HPP
#define CPPGRAD_VALUE_IMPL_HPP

#include <memory> // std::shared_ptr
#include <vector> // std::vector

namespace cppgrad {

// forward decl for topo
template <typename T>
class Value;

template <typename T>
using ValuePtr = std::shared_ptr<Value<T>>;

namespace impl {

    template <typename T>
    void build_topo(ValuePtr<T> v, std::vector<ValuePtr<T>>& topo)
    {
        if (v && !v->visited()) {
            auto [left, right] = v->_prev;

            v->visited() = true;

            build_topo(left, topo);
            build_topo(right, topo);

            topo.push_back(v);
        }
    }

    template <typename T>
    std::vector<ValuePtr<T>> build_topo(ValuePtr<T> v)
    {
        std::vector<ValuePtr<T>> topo;
        build_topo(v, topo);

        return topo;
    }

}

}

#endif