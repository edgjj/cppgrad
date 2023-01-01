// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_GENERIC_EXCEPTION_HPP
#define CPPGRAD_GENERIC_EXCEPTION_HPP

#include <stdexcept>

namespace cppgrad::exceptions {

struct GenericError : std::runtime_error {

    GenericError(const char* error)
        : std::runtime_error(error)
    {
    }
};

}

#endif