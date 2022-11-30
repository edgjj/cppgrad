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

    // default constructors / assignments
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    // atm these are just mock; actually we need to make some data copying
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    template <typename T>
    Tensor(T value)
    {
        // this is kinda weird to pass index and then cast it back to type
        *this = create<rtype_v<T>>({ 1 }, value);
    }

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

    template <DType DataType, typename DeviceType = CPU>
    static Tensor create(std::vector<size_t> shape = {},
        dtype_t<DataType> fill_value = dtype_t<DataType> { 0 },
        size_t alignment = alignof(dtype_t<DataType>))
    {
        auto type_size = sizeof(dtype_t<DataType>);

        size_t total_elements = std::reduce(shape.begin(), shape.end());
        auto strides = impl::make_strides(shape, type_size);

        std::align_val_t align { alignment };

        Device* device = new DeviceType();
        auto* chunk = device->allocate(total_elements * type_size, align);

        auto result = Tensor(chunk,
            std::move(shape),
            std::move(strides),
            align,
            device,
            DataType);

        result.fill(fill_value, total_elements);

        return result;
    }

    template <typename T>
    auto fill(T value, size_t count)
    {
        auto* byte_ptr = reinterpret_cast<std::byte*>(&value);
        _storage->_device->fill(data(), byte_ptr, dtype(), count);
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
    template <DType DataType>
    auto item()
    {
        if (empty()) {
            throw std::range_error("Tensor is empty.");
        }

        if (shape().size() > 1 || shape()[0] > 1) {
            throw std::range_error("Can only convert tensor of size 1 to a scalar.");
        }

        if (DataType != _storage->_type_id) {
            throw std::runtime_error("Requested type doesn't match content's type.");
        }

        using ResultType = dtype_t<DataType>;

        // use device instead
        return *reinterpret_cast<ResultType*>(_storage->_chunk);
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
            _storage->_device,
            _storage->_type_id };

        result._base = base_storage();

        return result;
    }

    /**
     * @brief Retrieves current data type stored in Tensor.
     *
     * @return DType type index
     */
    DType dtype() const
    {
        return base_storage()->_type_id;
    }

    /**
     * @brief Gets raw pointer to Tensor data.
     *
     * You must only use it if there's a REALLY good reason to do so.
     *
     * @return std::byte* raw data pointer
     */
    std::byte* data()
    {
        return _storage->_chunk;
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
            _storage->_device,
            _storage->_type_id };

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
            return *this;
        }

        if (CUDA::num_devices() == 0) {
            throw std::runtime_error("No available CUDA devices.");
        }

        // auto* new_device = new CUDA();
    }

    Tensor cpu()
    {
        // no need in cast
        if (!is_cuda_tensor()) {
            return *this;
        }

        // auto* new_device = new CPU();
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
        Device* device,
        DType type)
    {
        _storage = std::make_shared<impl::TensorData>(
            chunk,
            alignment,
            std::move(shape),
            std::move(strides),
            device,
            type);
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