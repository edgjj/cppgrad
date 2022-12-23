#ifndef CPPGRAD_TENSOR_IMPL_HPP
#define CPPGRAD_TENSOR_IMPL_HPP

#include <new> // std::align_val_t
#include <vector> // std::vector

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
            std::align_val_t alignment,
            std::vector<size_t>&& shape,
            std::vector<size_t>&& strides,
            Device* device,
            DType type)
            : _chunk(chunk)
            , _alignment(alignment)
            , _shape(std::move(shape))
            , _strides(std::move(strides))
            , _device(device)
            , _type_id(type)
        {
        }

        /**
         * @brief Pointer to row-major stored Tensor's data.
         */
        std::byte* _chunk;
        /**
         * @brief Pointer to Tensor's grad data;
         */
        std::byte* _grad_chunk;
        /**
         * @brief Tensor data alignment in bytes.
         */
        std::align_val_t _alignment;
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
        Device* _device { nullptr };

        /**
         * @brief ID of current type stored in Tensor.
         */
        DType _type_id { undefined };
    };

}

}

#endif