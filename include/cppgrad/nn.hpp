#ifndef CPPGRAD_NN_HPP
#define CPPGRAD_NN_HPP

namespace cppgrad
{

namespace nn
{

/*
	atm just a concept
*/

// look at this: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
class Module
{
public:
	using ModulePtr = Module*; // no ownership; btw could be made better thru class-local shit

	Module(ModulePtr parent = nullptr)
		: _parent (parent)
	{
		if (_parent)
		{
			// register itself
			// like _parent->register_child()
		}		
	}



private:

	void register_child(ModulePtr new_child)
	{
		_child.push_back(new_child);
	}

	std::vector<ModulePtr> _child;
	ModulePtr _parent;
};

/*

Sequential<
	Linear<InputSize, OutputSize>,
	Act::Tanh
	Linear<OutputSize, OutputSize2>,
	Act::Tanh,
	Linear<OutputSize2, 1>
	Act::Sigmoid
>

*/

}

}

#endif