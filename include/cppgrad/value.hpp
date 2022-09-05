#ifndef CPPGRAD_VALUE_HPP
#define CPPGRAD_VALUE_HPP

#include <utility>
#include <functional>
#include <cmath>
#include <vector>
#include <set>
#include <iostream>
#include <memory>
#include <optional>

namespace cppgrad 
{

// forward decl for topo
template <typename T>
class Value;

template <typename T>
using ValuePtr = std::shared_ptr<Value<T>>;

namespace util
{

template <typename T>
struct ValueComparator
{
	constexpr bool operator()(ValuePtr<T> a, ValuePtr<T> b) const
	{
		return a->_storage < b->_storage;
	}
};

template <typename T>
void build_topo(ValuePtr<T> v
	, std::set<ValuePtr<T>, util::ValueComparator<T>>& visited
	, std::vector<ValuePtr<T>>& topo)
{
	if (!v)
	{
		return;
	}

	if (visited.count(v) == 0)
	{
		std::cout << "[toposort] visiting: " << v << " op type: " << v->_op << " grad: " << v->_storage->grad << " value: " << v->_storage->val << '\n';
		std::cout << "\t" << " parent 1: " << v->_prev.first << " parent 2: " << v->_prev.second << '\n';

		visited.insert(v);

		auto [left, right] = v->_prev;

		build_topo(left, visited, topo);
		build_topo(right, visited, topo);

		topo.push_back(v);
	}
}

}

template <typename T>
class Value
{
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
	{}

	Value(T&& data, 
		BinaryPair parents = BinaryPair{},
		std::string op_name = "")
		: _storage(std::make_shared<ValueData<T>>(std::move(data), 0))
		, _prev{ parents }
		, _op{ op_name }
	{}

	// holy shit
	//using util::build_topo;
	friend void util::build_topo(ValuePtr<T> v
		, std::set<ValuePtr<T>, util::ValueComparator<T>>& visited
		, std::vector<ValuePtr<T>>& topo);

	// needed for set support
	friend struct util::ValueComparator<T>;

	Value<T> operator+(Value<T>& rhs)
	{
		auto self = _storage,
			other = rhs._storage;

		BinaryPair parents{ 
			std::make_shared<Value<T>>(*this), 
			std::make_shared<Value<T>>(rhs)
		};

		auto output = Value<T>(self->val + other->val
			, parents
			, "+");

		output._backward = [self, other, out = output._storage]() {
			self->grad += out->grad;
			other->grad += out->grad;
		};

		return output;
	}
	
	Value<T> operator*(Value<T>& rhs)
	{
		auto self = _storage,
			other = rhs._storage;

		BinaryPair parents{ 
			std::make_shared<Value<T>>(*this), 
			std::make_shared<Value<T>>(rhs)
		};

		auto output = Value<T>(self->val * other->val
			, parents
			, "*");

		output._backward = [self, other, out = output._storage]() {
			self->grad += other->val * out->grad;
			other->grad += self->val * out->grad;
		};

		return output;
	}
	
	Value<T> pow(Value<T>& rhs)
	{
		using std::pow;

		auto self = lhs._storage;
		
		BinaryPair parents{
			std::make_shared<Value<T>>(*this),
			std::nullptr_t{}
		};

		auto output = Value<T>(pow(self->val, rhs)
			, parents
			, "pow");

		output._backward = [self, rhs, out = output._storage]() {
			self->grad += rhs * pow(self->val, rhs - 1) * out->grad;
		};

		return output;
	}

	Value<T> relu()
	{
		auto self = _storage;

		BinaryPair parents{
			std::make_shared<Value<T>>(*this),
			std::nullptr_t{}
		};

		auto output = Value<T>(self->val < 0 ? 0 : self->val // this should be replaced with better op
			, parents
			, "relu");

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
		std::vector<ValuePtr<T>> topo;
		// we got std::hash specialization for this, so its ok
		std::set<ValuePtr<T>, util::ValueComparator<T>> visited;

		auto self = std::make_shared<Value<T>>(*this);
		util::build_topo(self, visited, topo);
		
		_storage->grad = T(1);

		for (auto rit = visited.rbegin(); rit != visited.rend(); rit++)
		{
			ValuePtr<T> cur = *rit;
			// check if _backward exists
			if (cur->_backward)
			{
				cur->_backward();

				std::cout << "[bward] visiting: " << cur << " op type: " << cur->_op << " grad: " << cur->_storage->grad << " value: " << cur->_storage->val << '\n';

			}
		}
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

	template <typename T>
	struct ValueData
	{
		ValueData(T&& _val, T&& _grad)
			: val(std::move(_val))
			, grad(std::move(_grad))
		{}

		T val;
		T grad;
	};

	using DataPtr = std::shared_ptr<ValueData<T>>;
	DataPtr _storage;

	BackwardFun _backward; // already nulled
	BinaryPair _prev; 

	std::string _op;
};

}

#endif