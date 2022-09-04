#ifndef CPPGRAD_VALUE_HPP
#define CPPGRAD_VALUE_HPP

#include <utility>
#include <functional>
#include <cmath>
#include <vector>
#include <set>

namespace cppgrad 
{

// forward decl for topo
template <typename T>
class Value;

template <typename T>
using ValuePtr = Value<T>*;

namespace util
{

template <typename T>
void build_topo(ValuePtr<T> v
	, std::set<ValuePtr<T>>& visited
	, std::vector<ValuePtr<T>>& topo)
{
	if (!v)
	{
		return;
	}

	if (visited.count(v) == 0)
	{
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
	using BackwardFun = std::function<void(T&, T&)>;

	/*
		We use this since all ops are binary or unary.
	*/
	using BinaryPair = std::pair<Value<T>*, Value<T>*>;

	Value(T&& data, 
		BackwardFun&& backward = BackwardFun{}, 
		BinaryPair parents = BinaryPair{})
		: _data{ std::move(data) }
		, _grad{ 0 }
		, _backward{ std::move(backward) }
		, _prev{ parents }
	{}

	// holy shit
	//using util::build_topo;
	friend void util::build_topo(ValuePtr<T> v
		, std::set<ValuePtr<T>>& visited
		, std::vector<ValuePtr<T>>& topo);

	friend Value<T> operator+(Value<T>& lhs, Value<T>& rhs)
	{
		BackwardFun fun = [&](T&, T& localGrad) {
			lhs._grad += localGrad;
			rhs._grad += localGrad;
		};

		BinaryPair parents{ &lhs, &rhs };
		auto output = Value<T>(lhs._data + rhs._data
			, std::move(fun)
			, parents);

		return output;
	}
	
	friend Value<T> operator*(Value<T>& lhs, Value<T>& rhs)
	{
		BackwardFun fun = [&](T&, T& localGrad) {
			lhs._grad += rhs._data * localGrad;
			rhs._grad += lhs._data * localGrad;
		};

		BinaryPair parents{ &lhs, &rhs };
		auto output = Value<T>(lhs._data * rhs._data
			, std::move(fun)
			, parents);

		return output;
	}
	
	Value<T> pow(Value<T>& rhs)
	{
		using std::pow;

		BackwardFun fun = [&](T&, T& localGrad) {
			_grad += rhs * pow(_data, rhs - 1) * localGrad;
		};

		BinaryPair parents{ this, nullptr };

		auto output = Value<T>(pow(_data, rhs)
			, std::move(fun)
			, parents);

		return output;
	}

	Value<T> relu()
	{
		BackwardFun fun = [&](T& localData, T& localGrad) {
			_grad += (localData > 0) * localGrad;
		};

		BinaryPair parents{ this, nullptr };

		auto output = Value<T>(_data < 0 ? 0 : _data
			, std::move(fun)
			, parents);

		return output;
	}

	/*!
	*	Calculates gradient using backprop. 
	*/
	void backward()
	{
		std::vector<ValuePtr<T>> topo;
		std::set<ValuePtr<T>> visited; // can't use set here, see below

		_grad = T( 1 );
		
		util::build_topo(this, visited, topo);
		
		for (auto rit = visited.rbegin(); rit != visited.rend(); rit++)
		{
			ValuePtr<T> cur = *rit;

			// check if _backward exists
			if (cur->_backward)
			{
				cur->_backward(cur->_data, cur->_grad);
			}
		}
	}

private:
	T _grad;
	T _data;

	BackwardFun _backward; // already nulled
	BinaryPair _prev; 
};

}

#endif