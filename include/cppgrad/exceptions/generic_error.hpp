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