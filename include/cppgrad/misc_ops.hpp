#ifndef CPPGRAD_VALUE_MISC_OPS_HPP
#define CPPGRAD_VALUE_MISC_OPS_HPP

namespace cppgrad
{

//// forward decl for toposort
//template <typename T>
//class Value;
//
//template <typename T>
//Value<T> operator-(Value<T>& lhs)
//{
//	return lhs * T(-1.0);
//}
//
//template <typename T>
//Value<T> operator-(Value<T>& lhs, Value<T>& rhs)
//{
//	return lhs + (-rhs);
//}
//
//template <typename T>
//Value<T> operator/(Value<T>& lhs, Value<T>& rhs)
//{
//	return lhs *rhs.pow(-1.0);
//}
//
//template <typename T>
//Value<T> operator+(T lhs, Value<T>& rhs)
//{
//	return Value<T>(std::move(lhs)) + rhs;
//}
//
//template <typename T>
//Value<T> operator*(T lhs, Value<T>& rhs)
//{
//	return Value<T>(std::move(lhs)) * rhs;
//}
//
//template <typename T>
//Value<T> operator/(T lhs, Value<T>& rhs)
//{
//	return Value<T>(std::move(lhs)) * rhs.pow(-1);
//}
//
//template <typename T>
//Value<T> operator/(Value<T>& lhs, T rhs)
//{
//	return lhs * Value<T>(std::move(lhs)).pow(-1);
//}
//
//template <typename T>
//Value<T> operator+= (Value<T>& lhs, Value<T>& rhs)
//{
//	return lhs + rhs;
//}

}

#endif