#ifndef CPPGRAD_TENSOR_HPP
#define CPPGRAD_TENSOR_HPP

#include <typeindex>	// std::type_index
#include <memory>		// std::shared_ptr
#include <vector>		// std::vector
#include <stdexcept>	// std::runtime_error

#include <numeric>

#include <cppgrad/config.hpp> // RTTI define

namespace cppgrad
{

class TensorItem
{

};

class Tensor
{
public:

	using DefaultType = float;

	template <typename T = DefaultType>
	T item()
	{
		if (_shape.size() == 0)
		{
			throw std::range_error("Tensor is empty.");
		}

		if (_shape.size() > 1 || _shape[0] > 1)
		{
			throw std::range_error("Can only convert tensor of size 1 to a scalar.");
		}

#ifdef CPPGRAD_HAS_RTTI
		if (typeid(T) != _type_holder)
		{
			throw std::runtime_error("Requested type doesn't match content's type.");
		}
#endif

		return *reinterpret_cast<T*>(_chunk);
	}

	Tensor operator[](size_t index)
	{
		std::vector<size_t> new_shape{ _shape.begin() + 1, _shape.end() };

		size_t reduced_shape = std::reduce(new_shape.begin(), new_shape.end());
		void* new_chunk = static_cast<char*>(_chunk) + index * reduced_shape * _type_size;

		Tensor result{ new_chunk, std::move(new_shape) };
		result._base = std::shared_ptr<Tensor>(this); // shared_ptr(this) but other way

		return result;
	}

private:

	/*
		private constructor for indexing
	*/
	Tensor(void* chunk, std::vector<size_t>&& shape) 
		: _chunk (chunk)
		, _shape (std::move(shape))
#ifdef CPPGRAD_HAS_RTTI
		, _type_holder(typeid(DefaultType))
#endif
	{
	}

	void* _chunk{ nullptr }; // void* cuz it can possibly located on GPU
	uint8_t _type_size{ 0 };
	uint8_t _type_alignment{ 0 };

	std::vector<size_t> _shape;

	std::shared_ptr<Tensor> _base; // for views - holds pointer to original chunk holder

#ifdef CPPGRAD_HAS_RTTI
	std::type_index _type_holder;
#endif

};


}

#endif 