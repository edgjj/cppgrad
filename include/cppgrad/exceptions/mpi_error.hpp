// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_MPI_EXCEPTION_HPP
#define CPPGRAD_MPI_EXCEPTION_HPP

#ifdef CPPGRAD_HAS_MPI

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cppgrad/distributed/error_string.hpp"

namespace cppgrad::exceptions {

namespace impl {
    inline std::string make_mpi_error_message(int err_code, const char* err_context)
    {
        std::stringstream s;
        s << err_context << ": " << distributed::impl::error_to_string(err_code) << ".";
        return s.str();
    }
}

struct MPIError : std::runtime_error {

    // no context error
    MPIError(int error_code)
        : std::runtime_error(impl::make_mpi_error_message(error_code, "MPI failure"))
    {
    }

    MPIError(int error_code, const char* context)
        : std::runtime_error(impl::make_mpi_error_message(error_code, context))
    {
    }
};

#define STRINGIFY(x) #x
#define TO_STRING(x) STRINGIFY(x)

// we should do same for CUDA i think
#define CPPGRAD_MPI_CHECK(op, ...)                      \
    if (auto code = op(__VA_ARGS__); code != MPI_SUCCESS) \
        throw exceptions::MPIError(code, "[ " __FILE__ ":" TO_STRING(__LINE__) " ]");

}

#endif

#endif