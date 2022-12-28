#ifndef CPPGRAD_DEFAULT_DEVICES_HPP
#define CPPGRAD_DEFAULT_DEVICES_HPP

#include "cppgrad/device/cpu/cpu.hpp"
#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/registry.hpp"
#include "cppgrad/device/tags.hpp"

namespace cppgrad {

namespace impl {
    bool register_defaults()
    {
        DeviceRegistry::get().register_device(RegistryEntry<CPU> { kCPU, "cpu" });
        DeviceRegistry::get().register_device(RegistryEntry<CUDA> { kCUDA, "cuda" });

        // cuda may self-register additional per-device executors
        return true;
    }

    struct Registrar {
        static bool _registered;
    };
}

// 'd register tha shit
bool impl::Registrar::_registered = impl::register_defaults();

}

#endif