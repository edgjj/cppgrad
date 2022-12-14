#ifndef CPPGRAD_VALUE_HPP
#define CPPGRAD_VALUE_HPP

#include <cmath> // std::pow
#include <functional> // std::function
#include <memory> // std::shared_ptr
#include <utility> // std::move
#include <vector> // std::vector

#include "cppgrad/autograd/impl.hpp" // build_topo
#include "cppgrad/autograd/misc_ops.hpp" // plus, minus, etc overloads

namespace cppgrad {

template <typename T>
class Value {
public:
    using BackwardFun = std::function<void()>;

    /*
            We use this since all ops are binary or unary.
    */
    using BinaryPair = std::pair<
        ValuePtr<T>,
        ValuePtr<T>>;

    Value()
        : _storage(std::make_shared<ValueData<T>>(0, 0))
    {
    }

    Value(T&& data,
        BinaryPair parents = BinaryPair {})
        : _storage(std::make_shared<ValueData<T>>(std::move(data), 0))
        , _prev { parents }
    {
    }

    template <typename Ty>
    friend void impl::build_topo(ValuePtr<Ty> v, std::vector<ValuePtr<Ty>>& topo);

    Value<T> operator+(const Value<T>& rhs) const
    {
        auto self = _storage,
             other = rhs._storage;

        BinaryPair parents {
            std::make_shared<Value<T>>(*this),
            std::make_shared<Value<T>>(rhs)
        };

        auto output = Value<T>(self->val + other->val, parents);

        output._backward = [self, other, out = output._storage]() {
            self->grad += out->grad;
            other->grad += out->grad;
        };

        return output;
    }

    Value<T> operator*(const Value<T>& rhs) const
    {
        auto self = _storage,
             other = rhs._storage;

        BinaryPair parents {
            std::make_shared<Value<T>>(*this),
            std::make_shared<Value<T>>(rhs)
        };

        auto output = Value<T>(self->val * other->val, parents);

        output._backward = [self, other, out = output._storage]() {
            self->grad += other->val * out->grad;
            other->grad += self->val * out->grad;
        };

        return output;
    }

    Value<T> pow(const Value<T>& rhs) const
    {
        using std::pow;

        auto self = _storage,
             other = rhs._storage;

        BinaryPair parents {
            std::make_shared<Value<T>>(*this),
            std::nullptr_t {}
        };

        auto output = Value<T>(pow(self->val, other->val), parents);

        output._backward = [self, other, out = output._storage]() {
            self->grad += other->val * pow(self->val, other->val - 1) * out->grad;
        };

        return output;
    }

    // move this outside tha class
    Value<T> relu() const
    {
        auto self = _storage;

        BinaryPair parents {
            std::make_shared<Value<T>>(*this),
            std::nullptr_t {}
        };

        auto output = Value<T>(self->val < 0 ? 0 : self->val, // this should be replaced with better op
            parents);

        output._backward = [self, out = output._storage]() {
            self->grad += (out->val > 0) * out->grad;
        };

        return output;
    }

    /*!
     *	Calculates gradient using backprop.
     */

    void backward()
    {
        auto self = std::make_shared<Value<T>>(*this);
        std::vector<ValuePtr<T>> topo = impl::build_topo(self);
        _storage->grad = T(1);

        for (auto rit = topo.rbegin(); rit != topo.rend(); rit++) {
            ValuePtr<T> cur = *rit;
            cur->_backward();
            // std::cout << "[bward] visiting " << cur << " grad: " << cur->grad() << " value: " << cur->data() << " \n";
        }

        for (auto& i : topo) {
            i->visited() = false;
        }
    }

    void zero_grad()
    {
        _backward = [] {};
        _prev = {};
        _storage->grad = T(0);
    }

    T& grad()
    {
        return _storage->grad;
    }

    T& data()
    {
        return _storage->val;
    }

private:
    bool& visited() // pathetic evil
    {
        return _storage->_visited;
    }

    template <typename Ty>
    struct ValueData {
        ValueData(Ty&& _val, Ty&& _grad)
            : val(std::move(_val))
            , grad(std::move(_grad))
        {
        }

        Ty val;
        Ty grad; // i think its possible to disable grad usage with NoGradGuard

        bool _visited = false;
    };

    using DataPtr = std::shared_ptr<ValueData<T>>;
    DataPtr _storage;

    BackwardFun _backward = [] {}; // already nulled
    BinaryPair _prev;
};

}

#endif