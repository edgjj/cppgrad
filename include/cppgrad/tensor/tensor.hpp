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

/*
    may need a .contiguous() type thing to make transposed matrices flat
*/
class Tensor {
public:
    using DefaultType = float;

    // default constructors / assignments
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    // atm these are just mock; actually we need to make some data copying
    Tensor& operator=(const Tensor&) = default;
    Tensor& operator=(Tensor&&) = default;

    // this ensures that _storage won't be empty
    Tensor()
        : Tensor(0)
    {
    }

    template <typename T>
    Tensor(T value)
    {
        // this is kinda weird to pass index and then cast it back to type
        *this = create<rtype_v<T>>({ 1 }, value);
    }

    template <typename T>
    Tensor(std::initializer_list<T> values)
    {
        *this = from_blob<rtype_v<T>>(const_cast<T*>(values.begin()),
            { values.size() });
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
        auto result = create_dirty<DeviceType>(shape, DataType, alignment);
        result.fill(fill_value, result.numel());

        return result;
    }

    /**
     * @brief Creates Tensor using memory blob with chosen shape.
     * Note: this function takes ownership of blob's data.
     *
     * @tparam DataType
     * @tparam DeviceType
     * @param blob
     * @param shape
     * @param alignment
     * @return Tensor
     */
    template <DType DataType, typename DeviceType = CPU>
    static Tensor from_blob(dtype_t<DataType>* blob,
        std::vector<size_t> shape = {},
        size_t alignment = alignof(dtype_t<DataType>))
    {
        if (blob == nullptr) {
            throw std::runtime_error("Caught nullptr blob.");
        }

        auto result = create_dirty<DeviceType>(shape, DataType, alignment);
        constexpr size_t type_sz = sizeof(dtype_t<DataType>);

        Device* device_ptr = result._storage->_device;

#ifdef CPPGRAD_HAS_CUDA
        if constexpr (std::is_same_v<DeviceType, CUDA>) {
            auto* cuda_device = dynamic_cast<CUDA*>(device_ptr);
            cuda_device->copy_from_host(reinterpret_cast<std::byte*>(blob), result.data(), type_sz * result.numel());

            return result;
        }
#endif

        device_ptr->copy(reinterpret_cast<std::byte*>(blob), result.data(), type_sz * result.numel());
        return result;
    }

    /**
     *
     * @brief Returns a Tensor of chosen shape, alignment and device, containing unitialized memory.
     *
     * Should be used for custom initialization routines.
     *
     * @tparam DataType
     * @tparam DeviceType
     * @param shape
     * @param alignment
     * @return Tensor
     */
    template <typename DeviceType = CPU>
    static Tensor create_dirty(std::vector<size_t> shape,
        DType type,
        size_t alignment)
    {
        auto type_size = dtype_size(type);

        size_t total_elements = std::reduce(shape.begin(), shape.end());
        auto strides = impl::make_strides(shape, type_size);

        std::align_val_t align { alignment };

        Device* device = new DeviceType();

        auto* chunk = device->allocate(total_elements * type_size, align);

        return Tensor(chunk,
            std::move(shape),
            std::move(strides),
            align,
            device,
            type);
    }

    /**
     * @brief Fills Tensor with desired type
     *
     * @tparam T
     * @param value
     * @param count
     */
    template <typename T>
    void fill(T value, size_t count)
    {
        if (rtype_v<T> != _storage->_type_id) {
            throw std::runtime_error("Requested type doesn't match Tensor's type.");
        }

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

#ifdef CPPGRAD_HAS_CUDA
        // we need transfer data from CUDA device first
        if (is_cuda_tensor()) {
            auto* cuda_device = dynamic_cast<CUDA*>(_storage->_device);
            dtype_t<DataType> data { 0 };

            cuda_device->copy_to_host(_storage->_chunk, reinterpret_cast<std::byte*>(&data), sizeof(data));

            return data;
        }
#endif

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

        // this is for case when a vector is stored inside tensor
        if (new_shape.empty()) {
            new_shape = { 1 };
            new_strides = { strides()[0] };
        }

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

    size_t numel() const noexcept
    {
        return std::reduce(shape().begin(), shape().end());
    }

    size_t nbytes() const noexcept
    {
        return numel() * dtype_size(dtype());
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
    /**
     * @brief Checks if current Tensor data is located on CUDA device.
     *
     */
    bool is_cuda_tensor() const
    {
        // we assume that there's always device attached to Tensor
        if (auto* ptr = dynamic_cast<CUDA*>(base_storage()->_device)) {
            return true;
        }

        return false;
    }

    /**
     * @brief Returns a new Tensor, having same data, but located on CUDA device.
     *
     * @return Tensor
     */
    Tensor cuda()
    {
        // safe pass if no CUDA devices
        if (is_cuda_tensor() || CUDA::num_devices() == 0) {
            return *this;
        }

        auto new_tensor = create_dirty<CUDA>(shape(), dtype(), (size_t)_storage->_alignment);
        auto* device = dynamic_cast<CUDA*>(new_tensor._storage->_device);

        device->copy_from_host(data(), new_tensor.data(), new_tensor.nbytes());

        return new_tensor;
    }

    /**
     * @brief Returns a new Tensor, having same data, but located on CPU memory.
     *
     * @return Tensor
     */
    Tensor cpu()
    {
        // no need in cast
        if (!is_cuda_tensor()) {
            return *this;
        }

        auto new_tensor = create_dirty<CPU>(shape(), dtype(), (size_t)_storage->_alignment);
        // get device of current tensor as its CUDA device
        auto* device = dynamic_cast<CUDA*>(_storage->_device);

        device->copy_to_host(new_tensor.data(), data(), nbytes());

        return new_tensor;
    }

#else
    // let it be just no-op if CUDA capatibilities are not built
    bool is_cuda_tensor() const
    {
        return false;
    }

    Tensor cpu()
    {
        return *this;
    }

    Tensor cuda()
    {
        return *this;
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
};
}

#endif