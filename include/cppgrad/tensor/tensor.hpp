#ifndef CPPGRAD_TENSOR_HPP
#define CPPGRAD_TENSOR_HPP

#include <algorithm> // std::is_sorted
#include <memory> // std::shared_ptr
#include <type_traits> // std::enable_if
#include <vector> // std::vector

#include <numeric>

#include "cppgrad/device/cpu/cpu.hpp"
#include "cppgrad/device/cuda/cuda.hpp"

#include "cppgrad/tensor/impl.hpp"
#include "cppgrad/tensor/impl_util.hpp"

// exceptions
#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/exceptions/index_error.hpp"
#include "cppgrad/exceptions/type_error.hpp"

namespace cppgrad {

/*
    may need a .contiguous() type thing to make transposed matrices flat
*/
class Tensor {
public:
    using DefaultType = float;

    // default constructors
    Tensor(const Tensor&) = default;
    Tensor(Tensor&&) = default;

    /**
     * @brief Copy assignment.
     * If left-hand Tensor is empty, shallow copy is done,
     * otherwise deep copy is
     *
     * @param other
     * @return Tensor&
     */
    Tensor& operator=(const Tensor& other)
    {
        // just return shallow copy for empty case
        if (empty()) {
            _storage = other._storage;
            _base = other._base;
            return *this;
        }

        CPPGRAD_CHECK_EQ(shape(), other.shape(),
            exceptions::GenericError,
            "Assign (move) shape mismatch");

        CPPGRAD_CHECK_EQ(dtype(), other.dtype(),
            exceptions::GenericError,
            "Assign (move) DType mismatch");

        CPPGRAD_CHECK_EQ(device()->type(), other.device()->type(),
            exceptions::GenericError,
            "Assign (move) Device type mismatch");

        if (is_contiguous() && other.is_contiguous()) {
            executor().copy(other.data(), data(), nbytes());
        } else {
            executor().strided_copy(other, *this);
        }

        return *this;
    }

    Tensor& operator=(Tensor&& other)
    {
        auto move_storage = [&]() -> Tensor& {
            _storage = std::move(other._storage);
            _base = std::move(other._base);
            return *this;
        };

        // just move if empty
        if (empty()) {
            return move_storage();
        }

        // check if assignment requirements satisfy
        CPPGRAD_CHECK_EQ(shape(), other.shape(),
            exceptions::GenericError,
            "Assign (move) shape mismatch");

        CPPGRAD_CHECK_EQ(dtype(), other.dtype(),
            exceptions::GenericError,
            "Assign (move) DType mismatch");

        // we can move safely if both are not views
        if (!is_view() && !other.is_view()) {
            return move_storage();
        }

        CPPGRAD_CHECK_EQ(device()->type(), other.device()->type(),
            exceptions::GenericError,
            "Assign (move) Device type mismatch");

        // fallback to copying;
        if (is_contiguous() && other.is_contiguous()) {
            executor().copy(other.data(), data(), nbytes());
        } else {
            executor().strided_copy(other, *this);
        }

        return *this;
    }

    // this ensures that _storage won't be empty
    Tensor()
        : Tensor(0)
    {
    }

    template <typename T, std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
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

    // ugly, is subject to change
    Tensor(std::initializer_list<Tensor> values)
    {
        CPPGRAD_CHECK_FALSE(values.begin()->empty(),
            exceptions::GenericError,
            "Initializer list initialization requires non-empty rows");

        auto& base_shape = values.begin()->shape();
        auto base_dtype = values.begin()->dtype();

        for (const auto& v : values) {
            if (!std::equal(base_shape.begin(), base_shape.end(), v.shape().begin())
                || base_dtype != v.dtype()) {
                throw exceptions::GenericError("Initializer list initialization requires all rows share same shape and DType");
            }
        }

        std::vector<size_t> shape { values.size() };
        shape.insert(shape.end(), base_shape.begin(), base_shape.end());

        auto align = (size_t)values.begin()->base_storage()->_alignment;
        *this = create_dirty(shape, base_dtype, align);

        for (size_t i = 0; i < values.size(); i++) {
            (*this)(i) = *(values.begin() + i);
        }
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
        CPPGRAD_CHECK_FALSE(blob == nullptr,
            exceptions::GenericError,
            "Caught nullptr blob.");

        auto result = create_dirty<DeviceType>(shape, DataType, alignment);
        constexpr size_t type_sz = sizeof(dtype_t<DataType>);

        /**
         *  We use HostToDevice there since DeviceType might be CUDA, or other non-CPU type.
         *  On CPU it's no-op.
         */
        result.executor().copy(reinterpret_cast<std::byte*>(blob), result.data(), type_sz * result.numel(), impl::HostToDevice);
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

        size_t total_elements = std::reduce(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
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
        CPPGRAD_CHECK_EQ(rtype_v<T>, _storage->_type_id,
            exceptions::TypeError,
            rtype_v<T>, _storage->_type_id);

        auto* byte_ptr = reinterpret_cast<std::byte*>(&value);
        executor().fill(*this, byte_ptr);
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
        CPPGRAD_CHECK_FALSE(empty(),
            exceptions::IndexError);

        CPPGRAD_CHECK_FALSE(shape().size() > 1 || shape()[0] > 1,
            exceptions::GenericError,
            "Can only convert tensor of size 1 to a scalar.");

        CPPGRAD_CHECK_EQ(DataType, _storage->_type_id,
            exceptions::TypeError, DataType, _storage->_type_id);

        using ResultType = dtype_t<DataType>;

        // we need transfer data from CUDA device first
        if (is_cuda_tensor()) {
            dtype_t<DataType> data { 0 };
            // copy_to_host
            executor().copy(_storage->_chunk, reinterpret_cast<std::byte*>(&data), sizeof(data), impl::DeviceToHost);

            return data;
        }

        return *reinterpret_cast<ResultType*>(_storage->_chunk);
    }

    /**
     * @brief Indexing operator. Returns tensor wrapping a view onto current tensor data.
     *
     * @param index
     * @return Tensor
     */

    template <typename... Indices>
    Tensor operator()(Indices... variadic_index)
    {
        constexpr size_t indices_size = sizeof...(Indices);
        size_t indices[indices_size] = { static_cast<size_t>(variadic_index)... };

        CPPGRAD_CHECK_FALSE(empty(),
            exceptions::IndexError);

        // not sure if it should be IndexError
        CPPGRAD_CHECK_FALSE(indices_size > shape().size(),
            exceptions::IndexError, indices_size, shape().size());

        auto& cur_strides = strides();
        auto& cur_shape = shape();

        std::vector<size_t> new_shape { cur_shape.begin() + indices_size, cur_shape.end() };
        std::vector<size_t> new_strides { cur_strides.begin() + indices_size, cur_strides.end() };

        // this is for case when a vector is stored inside tensor
        if (new_shape.empty()) {
            new_shape = { 1 };
            new_strides = { cur_strides[0] };
        }

        std::byte* new_chunk = _storage->_chunk;

        for (size_t i = 0; i < indices_size; i++) {
            CPPGRAD_CHECK_FALSE(indices[i] >= cur_shape[i],
                exceptions::IndexError, indices[i], cur_shape[i]);

            new_chunk += indices[i] * cur_strides[i];
        }

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
     * @brief Indexing operator. 1-dim operator() wrapper.
     *
     * @param index
     * @return Tensor
     */
    Tensor operator[](size_t index)
    {
        return (*this)(index);
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
    std::byte* data() const
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
        // right won't be evaluated if left was
        return !(bool)_storage || _storage->_shape.size() == 0 || _storage->_shape[0] == 0;
    }

    size_t numel() const noexcept
    {
        // note: 3rd arg explicit type is needed to avoid overflow
        return std::reduce(shape().begin(), shape().end(), size_t(1), std::multiplies<size_t>());
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

    Device* device() const
    {
        return base_storage()->_device;
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
        new_tensor.executor().copy(data(), new_tensor.data(), new_tensor.nbytes(), impl::HostToDevice);

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
        executor().copy(new_tensor.data(), data(), nbytes(), impl::DeviceToHost);

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

    /**
     * @brief Determines if Tensor is view onto other Tensor
     *
     */
    bool is_view() const
    {
        return (bool)_base;
    }

    /**
     * @brief Determines if Tensor's data stored contiguous.
     * Might be useful for working with transposed Tensors.
     *
     */
    bool is_contiguous() const
    {
        if (empty()) {
            return true;
        }

        // second condition required for 1dim tensors mostly
        return std::is_sorted(strides().rbegin(), strides().rend()) && *strides().rbegin() == dtype_size(dtype());
    }

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
        auto& actual_storage = is_view() ? _base : _storage;
        return actual_storage;
    }

    const std::shared_ptr<impl::TensorData>& base_storage() const
    {
        auto& actual_storage = is_view() ? _base : _storage;
        return actual_storage;
    }

    /**
     * @brief Utility method to get executor of attached device
     *
     * @return impl::Executor&
     */
    impl::Executor& executor()
    {
        return device()->get_executor();
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

private:
    std::shared_ptr<impl::TensorData> _storage;
    // duplicate for views
    std::shared_ptr<impl::TensorData> _base;
};
}

#endif