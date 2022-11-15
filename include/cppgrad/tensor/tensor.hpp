#ifndef CPPGRAD_TENSOR_HPP
#define CPPGRAD_TENSOR_HPP

#include <memory> // std::shared_ptr
#include <stdexcept> // std::runtime_error
#include <typeindex> // std::type_index
#include <vector> // std::vector

#include <numeric>

#include "cppgrad/device/registry.hpp"
#include "cppgrad/tensor/impl.hpp"

namespace cppgrad {

class Tensor {
public:
    using DefaultType = float;

    template <typename T = DefaultType>
    static Tensor create(std::vector<size_t> shape = {},
        T fill_value = T { 0 },
        size_t alignment = alignof(T),
        Device* device = DeviceRegistry::get("cpu"))
    {
        size_t total_elements = std::reduce(shape.begin(), shape.end());
        auto strides = impl::make_strides(shape, sizeof(T));

        std::align_val_t align { alignment };
        auto* chunk = device->allocate(total_elements * sizeof(T), align);

        return Tensor(chunk,
            std::move(shape),
            std::move(strides),
            align,
            device
#ifdef CPPGRAD_HAS_RTTI
            ,
            typeid(T)
#endif
        );
    }

    /**
     * @brief Queries scalar from tensor, given type T.
     *
     * \throws std::range_error Tensor is empty/has more than 1 element.
     * \throws std::runtime_error Queried and actual type are mismatched.
     *
     * @tparam T type to query
     * @return T scalar
     */
    template <typename T = DefaultType>
    T item()
    {
        if (empty()) {
            throw std::range_error("Tensor is empty.");
        }

        if (shape().size() > 1 || shape()[0] > 1) {
            throw std::range_error("Can only convert tensor of size 1 to a scalar.");
        }

#ifdef CPPGRAD_HAS_RTTI
        if (std::type_index(typeid(T)) != _storage->_type_holder) {
            throw std::runtime_error("Requested type doesn't match content's type.");
        }
#endif
        // use device instead
        return *reinterpret_cast<T*>(_storage->_chunk);
    }

    /**
     * @brief Lazy indexing operator. Returns tensor wrapping a view onto current tensor data.
     *
     * @param index
     * @return Tensor
     */
    Tensor operator[](size_t index)
    {
        if (empty()) {
            throw std::runtime_error("Trying to access empty Tensor.");
        }

        std::vector<size_t> new_shape { shape().begin() + 1, shape().end() };
        std::vector<size_t> new_strides { strides().begin() + 1, strides().end() };

        std::byte* new_chunk = _storage->_chunk + index * strides()[0];

        Tensor result { new_chunk,
            std::move(new_shape),
            std::move(new_strides),
            _storage->_alignment,
            _storage->_device
#ifdef CPPGRAD_HAS_RTTI
            ,
            _storage->_type_holder
#endif
        };
        result._base = _storage; // shared_ptr(this) but other way

        return result;
    }

    /**
     * @brief Tells whether tensor is empty or not.
     *
     * @return bool
     */
    bool empty() const noexcept
    {
        return _storage->_shape.size() == 0;
    }

    /**
     * @brief Return current tensor shape.
     *
     * @return const std::vector<size_t>&
     */
    const std::vector<size_t>& shape() const noexcept
    {
        return _storage->_shape;
    }

    /**
     * @brief Returns data indexing strides.
     *
     * @return const std::vector<size_t>&
     */
    const std::vector<size_t>& strides() const noexcept
    {
        return _storage->_strides;
    }

    /**
     * @brief Returns parent "base" tensor, in case if current tensor is view onto parent data.
     *        Otherwise returns shallow copy of current tensor.
     *
     * @return Tensor
     */
    Tensor base()
    {
        auto actual_storage = _base ? _base : _storage;
        return Tensor(std::move(actual_storage));
    }

    ~Tensor()
    {
        // check if we have parent TensorData attached
        auto& actual_storage = _base ? _base : _storage;

        if (actual_storage.use_count() == 1) {
            actual_storage->_device
                ->deallocate(actual_storage->_chunk, actual_storage->_alignment);
        }
    }

private:
    /**
     * @brief Construct a new Tensor object from pre-allocated chunk & other parameters.
     *        Main private ctor.
     *
     * @param chunk
     * @param shape
     * @param strides
     * @param alignment
     * @param device
     */
    Tensor(std::byte* chunk,
        std::vector<size_t>&& shape,
        std::vector<size_t>&& strides,
        std::align_val_t alignment,
        Device* device
#ifdef CPPGRAD_HAS_RTTI
        ,
        std::type_index type
#endif
    )
    {
        _storage = std::make_shared<impl::TensorData>(
            chunk,
            alignment,
            std::move(shape),
            std::move(strides),
            device
#ifdef CPPGRAD_HAS_RTTI
            ,
            type
#endif
        );
    }

    /**
     * @brief Construct a new Tensor object from parent's TensorData.
     *
     * @param base_storage Parent's TensorData.
     */
    Tensor(std::shared_ptr<impl::TensorData> base_storage)
        : _storage(std::move(base_storage))
    {
    }

    std::shared_ptr<impl::TensorData> _storage;
    // duplicate for views
    std::shared_ptr<impl::TensorData> _base;

    /*
        we do need to make following things:
                per-dimension strides (basically a step for which we should advance for chosen dimension)
                stride-based indexing

        ... PROFIT!
    */
};
}

#endif