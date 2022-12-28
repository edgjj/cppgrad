#ifndef CPPGRAD_DEFAULT_DEVICES_HPP
#define CPPGRAD_DEFAULT_DEVICES_HPP

#include "cppgrad/device/cpu/cpu.hpp"
#include "cppgrad/device/cuda/cuda.hpp"
#include "cppgrad/device/registry.hpp"
#include "cppgrad/device/tags.hpp"
#include "cppgrad/tensor/tensor.hpp"

namespace cppgrad {

namespace impl {
    bool register_defaults()
    {
        DeviceRegistry::get().register_device(RegistryEntry<CPU> { kCPU, "cpu" });
#ifdef CPPGRAD_HAS_CUDA
        // cuda may self-register additional per-device executors
        DeviceRegistry::get().register_device(RegistryEntry<CUDA> { kCUDA, "cuda" });
#endif
        return true;
    }

    struct Registrar {
        // we need Tensor there since backends require Tensor symbols
        // if we don't use Tensor there, and Tensor isn't used anywhere in program -> linking 'd fail
        static Tensor _tensor;
        static bool _registered;
    };
}

// 'd register tha shit
bool impl::Registrar::_registered = impl::register_defaults();
Tensor impl::Registrar::_tensor = Tensor();

}

#endif