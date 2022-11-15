#ifndef CPPGRAD_TENSOR_IMPL_HPP
#define CPPGRAD_TENSOR_IMPL_HPP

#include <new> // std::align_val_t
#include <vector> // std::vector

#include "cppgrad/config.hpp" // RTTI define

#ifdef CPPGRAD_HAS_RTTI
#include <typeindex> // std::type_index
#endif

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
            Device* device
#ifdef CPPGRAD_HAS_RTTI
            ,
            std::type_index type
#endif
            )
            : _chunk(chunk)
            , _alignment(alignment)
            , _shape(std::move(shape))
            , _strides(std::move(strides))
            , _device(device)
#ifdef CPPGRAD_HAS_RTTI
            , _type_holder(type)
#endif
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
         * @brief Current type stored in Tensor. Disabled if no RTTI enabled.
         */
#ifdef CPPGRAD_HAS_RTTI
        std::type_index _type_holder;
#endif
    };

}

}

#endif