#ifndef CPPGRAD_DEVICE_REGISTRY_HPP
#define CPPGRAD_DEVICE_REGISTRY_HPP

#include "cppgrad/device/device.hpp"

#include <string>
#include <unordered_map>

namespace cppgrad {

struct DeviceRegistry {

    using DeviceMap = std::unordered_map<std::string, Device*>;

    template <typename T>
    friend class DeviceRegistrar;

    static Device* get(const std::string& name)
    {
        return devices().at(name);
    }

private:
    static bool reg(const std::string& name, Device* ptr)
    {
        if (devices().count(name) == 0) {
            devices()[name] = ptr;
            return true;
        }

        return false;
    }

    static DeviceMap& devices()
    {
        static DeviceMap _devices;

        return _devices;
    }
};

template <typename T>
struct DeviceRegistrar {
    explicit DeviceRegistrar(const std::string& name)
    {
        static T device;
        DeviceRegistry::reg(name, &device);
    }
};

#define REGISTER_DEVICE(type, name)                    \
    namespace {                                        \
        static DeviceRegistrar<type> device_reg(name); \
    }

}

#endif