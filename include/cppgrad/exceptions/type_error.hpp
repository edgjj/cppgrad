// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_TYPE_EXCEPTION_HPP
#define CPPGRAD_TYPE_EXCEPTION_HPP

#include "cppgrad/tensor/typing.hpp"
#include <sstream>
#include <stdexcept>
#include <string>

namespace cppgrad::exceptions {

namespace impl {
    inline std::string make_type_message(DType requested_type, DType real_type)
    {
        std::stringstream s;
        s << "Requested type doesn't match Tensor's type. ";
        s << "Type: " << dtype_name(requested_type);
        s << ". Tensor's type: " << dtype_name(real_type) << ".";

        return s.str();
    }
}

struct TypeError : std::runtime_error {

    TypeError(DType requested_type, DType real_type)
        : std::runtime_error(impl::make_type_message(requested_type, real_type))
    {
    }
};

}

#endif