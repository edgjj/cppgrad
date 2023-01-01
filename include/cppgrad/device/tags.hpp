// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_DEVICE_TAGS_HPP
#define CPPGRAD_DEVICE_TAGS_HPP

#include <cstddef>

namespace cppgrad {

enum DefaultDeviceTag : size_t {
    kCPU,
    kCUDA
};

using DeviceTag = size_t;

}

#endif