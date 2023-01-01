// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_TENSOR_IMPL_HPP
#define CPPGRAD_TENSOR_IMPL_HPP

#include <vector> // std::vector

#include "cppgrad/autograd/context.hpp"
#include "cppgrad/device/tags.hpp"
#include "cppgrad/tensor/typing.hpp"

namespace cppgrad {

struct Device;

namespace impl {

    /**
     * @brief Data object encapsulating all Tensor data.
     *        Required for lazy copying semantics.
     *
     */
    struct TensorData {
        TensorData(std::byte* chunk,
            const std::vector<size_t>& shape,
            const std::vector<size_t>& strides,
            DeviceTag device,
            DType type)
            : _chunk(chunk)
            , _shape(shape)
            , _strides(strides)
            , _device(device)
            , _type_id(type)
        {
        }

        /**
         * @brief Pointer to row-major stored Tensor's data.
         */
        std::byte* _chunk;
        /**
         * @brief Tensor shape.
         */
        std::vector<size_t> _shape;
        /**
         * @brief Tensor data strides in bytes.
         */
        std::vector<size_t> _strides;

        /**
         * @brief Device (allocator) Tensor works with.
         */
        DeviceTag _device;

        /**
         * @brief ID of current type stored in Tensor.
         */
        DType _type_id { undefined };

        std::unique_ptr<autograd::AutogradInterface> _autograd_context;
    };

}

}

#endif