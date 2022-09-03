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

namespace util
{

template <typename T>
void build_topo(Value<T>& v
	, std::set<Value<T>>& visited
	, std::vector<Value<T>>& topo)
{
	if (visited.count(v) == 0)
	{
		visited.insert(v);
		for (auto& child : v._prev)
		{
			build_topo(child, visited, topo);
		}
		topo.push_back(v);
	}
}

}

template <typename T>
class Value
{
public:
	using BackwardFun = std::function<void(T&, T&)>;

	Value(T&& data, BackwardFun&& backward = BackwardFun{})
		: _data{ std::move(data) }
		, _grad{ 0 }
		, _backward{ std::move(backward) }
	{}

	// holy shit
	friend void util::build_topo(Value<T>& v
		, std::set<Value<T>>& visited
		, std::vector<Value<T>>& topo);

	friend Value<T> operator+(Value<T>& lhs, Value<T>& rhs)
	{
		BackwardFun fun = [&](T&, T& localGrad) {
			lhs._grad += localGrad;
			rhs._grad += localGrad;
		};

		auto output = Value<T>(lhs._data + rhs._data
			, std::move(fun));

		return output;
	}
	
	friend Value<T> operator*(Value<T>& lhs, Value<T>& rhs)
	{
		BackwardFun fun = [&](T&, T& localGrad) {
			lhs._grad += rhs._data * localGrad;
			rhs._grad += lhs._data * localGrad;
		};

		auto output = Value<T>(lhs._data * rhs._data
			, std::move(fun));

		return output;
	}
	
	Value<T> pow(Value<T>& rhs)
	{
		using std::pow;

		BackwardFun fun = [&](T&, T& localGrad) {
			_grad += rhs * pow(_data, rhs - 1) * localGrad;
		};

		auto output = Value<T>(pow(_data, rhs)
			, std::move(fun));

		return output;
	}

	Value<T> relu()
	{
		BackwardFun fun = [&](T& localData, T& localGrad) {
			_grad += (localData > 0) * localGrad;
		};

		auto output = Value<T>(_data < 0 ? 0 : _data
			, std::move(fun));

		return output;
	}

	/*!
	*	Calculates gradient using backprop. 
	*/
	void backward()
	{
		using Val = Value<T>;

		std::vector<Val> topo;
		std::set<Val> visited;

		_grad = 1;
		
		
		for (auto rit = visited.rbegin(); rit != visited.rend(); rit++)
		{
			Val& cur = *rit;
			cur._backward(cur._data, cur._grad);
		}
	}

private:
	T _grad;
	T _data;

	BackwardFun _backward; // already nulled
};

}

#endif