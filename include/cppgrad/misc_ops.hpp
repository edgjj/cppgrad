#ifndef CPPGRAD_VALUE_MISC_OPS_HPP
#define CPPGRAD_VALUE_MISC_OPS_HPP

namespace cppgrad
{

// forward decl
template <typename T>
class Value;

/*
	negate / diff operators
*/
template <typename T>
Value<T> operator-(const Value<T>& lhs) // _neg_
{
	return lhs * T(-1.0);
}

template <typename T>
Value<T> operator-(const Value<T>& lhs, const Value<T>& rhs) // _sub_
{
	return lhs + rhs * T(-1.0);
}

template <typename T>
Value<T> operator-(const Value<T>& lhs, T&& rhs) // _sub_ for T being rhs
{
	return lhs - Value<T>(std::forward<T>(rhs));
}

template <typename T>
Value<T> operator-(T&& lhs, const Value<T>& rhs) // _rsub_
{
	return Value<T>(std::forward<T>(lhs)) - rhs;
}

/*
	multiply operators to cast from T
*/


template <typename T>
Value<T> operator*(const Value<T>& lhs, T&& rhs) // _mul for T being rhs
{
	return lhs * Value<T>(std::forward<T>(rhs));
}

template <typename T>
Value<T> operator*(T&& lhs, const Value<T>& rhs) // _rmul_
{
	return Value<T>(std::forward<T>(lhs)) * rhs;
}

/*
	division operators
*/
template <typename T>
Value<T> operator/(const Value<T>& lhs, const Value<T>& rhs) // _truediv_
{
	return lhs * rhs.pow(-1.0);
}

template <typename T>
Value<T> operator/(const Value<T>& lhs, T&& rhs) // _truediv_ for T being rhs
{
	return lhs * Value<T>(std::forward<T>(rhs)).pow(-1);
}

template <typename T>
Value<T> operator/(T&& lhs, const Value<T>& rhs) // _rtruediv_
{
	return Value<T>(std::forward<T>(lhs)) * rhs.pow(-1);
}

/*
	xtra add support
*/
template <typename T>
Value<T> operator+(T&& lhs, const Value<T>& rhs) // _radd_
{
	return Value<T>(std::forward<T>(lhs)) + rhs;
}

}

#endif