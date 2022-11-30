#ifndef CPPGRAD_TENSOR_IMPL_HPP
#define CPPGRAD_TENSOR_IMPL_HPP

#include <new> // std::align_val_t
#include <vector> // std::vector

#include "cppgrad/tensor/typing.hpp"

namespace cppgrad {

struct Device;

namespace impl {

    /**
     * @brief Internal function to make Tensor strides.
     *
     * @param shape Requested tensor shape
     * @param type_size Requested tensor type size in bytes
     * @return std::vector<size_t> Strides
     */
    std::vector<size_t> make_strides(std::vector<size_t> shape, size_t type_size)
    {
        std::vector<size_t> strides(shape.size());
        size_t accum = type_size;

        auto it_stride = strides.rbegin();

        for (auto it = shape.rbegin(); it != shape.rend(); it++) {
            *it_stride = accum;
            accum *= *it;
            it_stride++;
        }

        return strides;
    }

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
            cppgrad_id type)
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
        unsigned _type_id { 0xFFFF };
    };

}

}

#endif