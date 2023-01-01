// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_INDEX_EXCEPTION_HPP
#define CPPGRAD_INDEX_EXCEPTION_HPP

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cppgrad::exceptions {

namespace impl {
    inline std::string make_index_message(size_t index, size_t dimension_size)
    {
        std::stringstream s;
        s << "Index " << std::to_string(index) << " is out of bounds with size: ";
        s << std::to_string(dimension_size) << ".";

        return s.str();
    }
}

struct IndexError : std::runtime_error {

    // empty error
    IndexError()
        : std::runtime_error("Tensor is empty.")
    {
    }

    // indexing error
    IndexError(size_t index, size_t dimension_size)
        : std::runtime_error(impl::make_index_message(index, dimension_size))
    {
    }
};

}

#endif