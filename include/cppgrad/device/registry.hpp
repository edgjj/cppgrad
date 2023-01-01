// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_DEVICE_REGISTRY_HPP
#define CPPGRAD_DEVICE_REGISTRY_HPP

#include "cppgrad/device/device.hpp"
#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/device/tags.hpp"

#include <any>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

namespace cppgrad {

template <typename DeviceType>
struct RegistryEntry {
    size_t numeric_tag;
    std::string string_tag;
};

struct DeviceRegistry {

    template <typename DeviceType>
    void register_device(RegistryEntry<DeviceType> entry)
    {
        if (_numeric_index.count(entry.numeric_tag) != 0 || _string_index.count(entry.string_tag) != 0) {
            throw exceptions::GenericError("There's already device with this tag");
        }

        _devices.emplace_back(new DeviceType());

        _numeric_index[entry.numeric_tag]
            = _string_index[entry.string_tag]
            = _devices.size() - 1;

        _string_to_numeric_index[entry.string_tag] = entry.numeric_tag;
    }

    Device& by_numeric(size_t tag)
    {
        // throws if not found
        return *_devices[_numeric_index.at(tag)];
    }

    Device& by_string(const std::string& tag)
    {
        // throws if not found
        return *_devices[_string_index.at(tag)];
    }

    size_t get_numeric_tag(const std::any& tag)
    {
        if (auto* p = std::any_cast<std::string>(&tag)) {
            return _string_to_numeric_index.at(*p);
        }
        else if (auto* p = std::any_cast<size_t>(&tag)) {
            return *p;
        }
        else if (auto* p = std::any_cast<DefaultDeviceTag>(&tag)) { // its different types !
            return *p;
        }
        else {
            throw exceptions::GenericError("Invalid any tag shipped to by_any; Allowed: std::string, size_t");
        }
    }

    static DeviceRegistry& get()
    {
        static DeviceRegistry registry;
        return registry;
    }

private:
    DeviceRegistry() = default;

    DeviceRegistry(const DeviceRegistry&) = delete;
    DeviceRegistry(DeviceRegistry&&) = delete;

    std::unordered_map<size_t, size_t> _numeric_index;
    std::unordered_map<std::string, size_t> _string_index;

    std::unordered_map<std::string, size_t> _string_to_numeric_index;

    std::vector<std::unique_ptr<Device>> _devices;
};

}

#endif