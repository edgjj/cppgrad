#ifndef CPPGRAD_DEVICE_HPP
#define CPPGRAD_DEVICE_HPP

#include <string>
#include <unordered_map>

namespace cppgrad {

struct Device;

namespace impl {
    static std::unordered_map<std::string, Device*> devices;
}

enum class DeviceType {
    CPU = 2,
    CUDA = 4,
};

struct Device {

    virtual std::byte* allocate(std::size_t count, std::align_val_t alignment) = 0;
    virtual void deallocate(std::byte* ptr, std::align_val_t alignment) = 0;
    virtual void copy(std::byte* from, std::byte* to, std::size_t count) = 0;

    virtual DeviceType type() = 0;

    static Device& get(const std::string& name)
    {
        return *impl::devices.at(name);
    }
};

struct DeviceRegistrar {
    DeviceRegistrar(const std::string& name, Device* ptr)
    {
        if (impl::devices.count(name) == 0) {
            impl::devices[name] = ptr;
        }
    }
};

#define REGISTER_DEVICE(name, ptr)                    \
    namespace {                                       \
        static DeviceRegistrar device_reg(name, ptr); \
    }

}

#endif