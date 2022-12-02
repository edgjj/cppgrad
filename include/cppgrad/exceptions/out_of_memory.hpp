#ifndef CPPGRAD_OOM_EXCEPTION_HPP
#define CPPGRAD_OOM_EXCEPTION_HPP

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cppgrad::exceptions {

namespace impl {
    inline std::string make_oom_message(std::string_view device_type, size_t nbytes_failed)
    {
        std::stringstream s;
        s << "[ " << device_type << " ]";
        s << " Device out of memory. Tried to allocate: " << std::to_string(nbytes_failed) << " bytes.";

        return s.str();
    }

    inline std::string make_oom_message(std::string_view device_type, size_t nbytes_failed, size_t nbytes_available)
    {
        std::stringstream s(make_oom_message(device_type, nbytes_failed));
        s << " Available memory: " << std::to_string(nbytes_available) + " bytes.";

        return s.str();
    }
}

struct OutOfMemoryError : std::runtime_error {

    OutOfMemoryError(std::string_view device_type, size_t nbytes_failed)
        : std::runtime_error(impl::make_oom_message(device_type, nbytes_failed))
    {
    }

    OutOfMemoryError(std::string_view device_type, size_t nbytes_failed, size_t nbytes_available)
        : std::runtime_error(impl::make_oom_message(device_type, nbytes_failed, nbytes_available))
    {
    }
};

}

#endif