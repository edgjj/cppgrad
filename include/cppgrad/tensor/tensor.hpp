#ifndef CPPGRAD_TENSOR_HPP
#define CPPGRAD_TENSOR_HPP

#include <memory> // std::shared_ptr
#include <stdexcept> // std::runtime_error
#include <typeindex> // std::type_index
#include <vector> // std::vector

#include <numeric>

#include "cppgrad/device/cpu/cpu.hpp"
#include "cppgrad/device/cuda/cuda.hpp"

#include "cppgrad/tensor/impl.hpp"

namespace cppgrad {

class Tensor {
public:
    using DefaultType = float;

    /**
     * @brief Returns a Tensor of chosen shape, alignment and device, filled with some value.
     *
     * @tparam T
     * @param shape
     * @param fill_value
     * @param alignment
     * @param device
     * @return Tensor
     */

    template <typename DeviceType, typename T = DefaultType>
    static Tensor create(std::vector<size_t> shape = {},
        T fill_value = T { 0 },
        size_t alignment = alignof(T))
    {
        size_t total_elements = std::reduce(shape.begin(), shape.end());
        auto strides = impl::make_strides(shape, sizeof(T));

        std::align_val_t align { alignment };
        auto* chunk = device.allocate(total_elements * sizeof(T), align);
        auto* device = new DeviceType();

        return Tensor(chunk,
            std::move(shape),
            std::move(strides),
            align,
            &device
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

        result._base = base_storage();

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
        return Tensor(base_storage());
    }

    /**
     * @brief Makes transposed view to Tensor.
     *
     * @return Tensor transposed Tensor
     */
    Tensor T()
    {
        std::vector<size_t> new_shape { shape().rbegin(), shape().rend() };
        std::vector<size_t> new_strides { strides().rbegin(), strides().rend() };

        Tensor result { _storage->_chunk,
            std::move(new_shape),
            std::move(new_strides),
            _storage->_alignment,
            _storage->_device
#ifdef CPPGRAD_HAS_RTTI
            ,
            _storage->_type_holder
#endif
        };

        result._base = base_storage();

        return result;
    }

#ifdef CPPGRAD_HAS_CUDA
    bool is_cuda_tensor() const
    {
        if (!(base_storage()->_device)) {
            return false;
        }

        if (auto* ptr = dynamic_cast<CUDA*>(base_storage()->_device)) {
            return true;
        }

        return false;
    }

    Tensor cuda()
    {
        if (is_cuda_tensor()) {
            return;
        }

        if (CUDA::num_devices() == 0) {
            throw std::runtime_error("No available CUDA devices.");
        }

        auto* new_device = new CUDA();
    }

    Tensor cpu()
    {
        // no need in cast
        if (!is_cuda_tensor()) {
            return;
        }

        auto* new_device = new CPU();
    }

#endif

    ~Tensor()
    {
        // check if we have parent TensorData attached
        auto& actual_storage = base_storage();

        if (actual_storage.use_count() == 1) {
            actual_storage->_device
                ->deallocate(actual_storage->_chunk, actual_storage->_alignment);

            delete actual_storage->_device;
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
     * @brief Utility method to get actual base TensorData storage pointer.
     *
     * @return std::shared_ptr<impl::TensorData>& Parent's TensorData
     */
    std::shared_ptr<impl::TensorData>& base_storage()
    {
        auto& actual_storage = _base ? _base : _storage;
        return actual_storage;
    }

    const std::shared_ptr<impl::TensorData>& base_storage() const
    {
        auto& actual_storage = _base ? _base : _storage;
        return actual_storage;
    }

    /**
     * @brief Construct a new Tensor object from parent's TensorData.
     *
     * @param base_storage Parent's TensorData
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