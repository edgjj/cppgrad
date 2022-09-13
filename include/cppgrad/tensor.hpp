#ifndef CPPGRAD_TENSOR_HPP
#define CPPGRAD_TENSOR_HPP

#include <typeinfo>

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

	}

private:

	void* _chunk; // void* cuz it can possibly located on GPU
	std::type_info _type_holder;
	
};


}

#endif 