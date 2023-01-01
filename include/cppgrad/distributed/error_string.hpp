// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_DISTRIBUTED_ERROR_STRING_HPP
#define CPPGRAD_DISTRIBUTED_ERROR_STRING_HPP

#ifdef CPPGRAD_HAS_MPI

#include <mpi.h>
#include <string>

namespace cppgrad::distributed::impl {

inline std::string error_to_string(int error_code)
{
    char err_buffer[MPI_MAX_ERROR_STRING];
    int len = 0;

    MPI_Error_string(error_code, err_buffer, &len);

    return std::string(err_buffer, len);
}

}

#endif

#endif