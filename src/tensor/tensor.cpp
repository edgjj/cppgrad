#include <algorithm> // std::is_sorted

#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/exceptions/type_error.hpp"
#include "cppgrad/tensor/tensor.hpp"

#include "cppgrad/device/cuda/cuda.hpp"

namespace cppgrad {

Tensor& Tensor::operator=(const Tensor& other)
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

    CPPGRAD_CHECK_EQ(device().type(), other.device().type(),
        exceptions::GenericError,
        "Assign (move) Device type mismatch");

    if (is_contiguous() && other.is_contiguous()) {
        executor().copy(other.data(), data(), nbytes());
    } else {
        executor().strided_copy(other, *this);
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other)
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

    CPPGRAD_CHECK_EQ(device().type(), other.device().type(),
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

Tensor::Tensor()
    : Tensor(0)
{
}

template <typename Type, std::enable_if_t<std::is_arithmetic_v<Type>>*>
Tensor::Tensor(Type value)
{
    // this is kinda weird to pass index and then cast it back to type
    *this = create<rtype_v<Type>>({ 1 }, value);
}

template <typename Type>
Tensor::Tensor(std::initializer_list<Type> values)
{
    *this = from_blob<rtype_v<Type>>(const_cast<Type*>(values.begin()),
        { values.size() });
}

// ugly, is subject to change
Tensor::Tensor(std::initializer_list<Tensor> values)
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
    *this = create_dirty(shape, base_dtype, align, new CPU());

    for (size_t i = 0; i < values.size(); i++) {
        (*this)(i) = *(values.begin() + i);
    }
}

template <typename Type>
void Tensor::fill(Type value, size_t count)
{
    CPPGRAD_CHECK_EQ(rtype_v<Type>, _storage->_type_id,
        exceptions::TypeError,
        rtype_v<Type>, _storage->_type_id);

    auto* byte_ptr = reinterpret_cast<std::byte*>(&value);
    executor().fill(*this, byte_ptr);
}

template <DType DataType>
dtype_t<DataType> Tensor::item()
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

Tensor Tensor::operator[](size_t index)
{
    return (*this)(index);
}

DType Tensor::dtype() const
{
    return base_storage()->_type_id;
}

std::byte* Tensor::data()
{
    return _storage->_chunk;
}

const std::byte* Tensor::data() const
{
    return _storage->_chunk;
}

bool Tensor::empty() const noexcept
{
    // right won't be evaluated if left was
    return !(bool)_storage || _storage->_shape.size() == 0 || _storage->_shape[0] == 0;
}

size_t Tensor::numel() const noexcept
{
    // note: 3rd arg explicit type is needed to avoid overflow
    return std::reduce(shape().begin(), shape().end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::nbytes() const noexcept
{
    return numel() * dtype_size(dtype());
}

const std::vector<size_t>& Tensor::shape() const noexcept
{
    return _storage->_shape;
}

const std::vector<size_t>& Tensor::strides() const noexcept
{
    return _storage->_strides;
}

Tensor Tensor::base()
{
    return Tensor(base_storage());
}

Tensor Tensor::T()
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

Device& Tensor::device() const
{
    return *base_storage()->_device;
}

#ifdef CPPGRAD_HAS_CUDA
bool Tensor::is_cuda_tensor() const
{
    // we assume that there's always device attached to Tensor
    if (auto* ptr = dynamic_cast<CUDA*>(base_storage()->_device)) {
        return true;
    }

    return false;
}

Tensor Tensor::cuda()
{
    // safe pass if no CUDA devices
    if (is_cuda_tensor() || CUDA::num_devices() == 0) {
        return *this;
    }

    auto new_tensor = create_dirty(shape(), dtype(), (size_t)_storage->_alignment, new CUDA());
    new_tensor.executor().copy(data(), new_tensor.data(), new_tensor.nbytes(), impl::HostToDevice);

    return new_tensor;
}

Tensor Tensor::cpu()
{
    // no need in cast
    if (!is_cuda_tensor()) {
        return *this;
    }

    auto new_tensor = create_dirty(shape(), dtype(), (size_t)_storage->_alignment, new CPU());
    executor().copy(new_tensor.data(), data(), nbytes(), impl::DeviceToHost);

    return new_tensor;
}

#else
// let it be just no-op if CUDA capatibilities are not built
bool Tensor::is_cuda_tensor() const
{
    return false;
}

Tensor Tensor::cpu()
{
    return *this;
}

Tensor Tensor::cuda()
{
    return *this;
}
#endif

Tensor Tensor::clone() const
{
    auto new_tensor = create_dirty(shape(), dtype(), get_align(), device().clone());
    // copy strides to be the same
    new_tensor._storage->_strides = _storage->_strides;
    executor().copy(data(), new_tensor.data(), nbytes());

    return new_tensor;
}

Tensor Tensor::contiguous() const
{
    // just return copy
    if (is_contiguous()) {
        return *this;
    }

    auto new_tensor = create_dirty(shape(), dtype(), get_align(), device().clone());
    executor().strided_copy(*this, new_tensor);

    return new_tensor;
}

bool Tensor::is_view() const
{
    return (bool)_base;
}

bool Tensor::is_contiguous() const
{
    if (empty()) {
        return true;
    }

    // second condition required for 1dim tensors mostly
    return std::is_sorted(strides().rbegin(), strides().rend()) && *strides().rbegin() == dtype_size(dtype());
}

size_t Tensor::get_align() const
{
    return (size_t)base_storage()->_alignment;
}

Tensor::~Tensor()
{
    // check if we have parent TensorData attached
    auto& actual_storage = base_storage();

    if (actual_storage.use_count() == 1) {
        actual_storage->_device
            ->deallocate(actual_storage->_chunk, actual_storage->_alignment);

        delete actual_storage->_device;
    }
}

Tensor::Tensor(std::byte* chunk,
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

std::shared_ptr<impl::TensorData>& Tensor::base_storage()
{
    auto& actual_storage = is_view() ? _base : _storage;
    return actual_storage;
}

const std::shared_ptr<impl::TensorData>& Tensor::base_storage() const
{
    auto& actual_storage = is_view() ? _base : _storage;
    return actual_storage;
}

impl::Executor& Tensor::executor() const
{
    return device().get_executor();
}

Tensor::Tensor(std::shared_ptr<impl::TensorData> base_storage)
    : _storage(std::move(base_storage))
{
}

// template instantiations go here

/* Tensor::Tensor(Type value) */
template Tensor::Tensor(dtype_t<u32> value);
template Tensor::Tensor(dtype_t<u64> value);
template Tensor::Tensor(dtype_t<i32> value);
template Tensor::Tensor(dtype_t<i64> value);
template Tensor::Tensor(dtype_t<f32> value);
template Tensor::Tensor(dtype_t<f64> value);

/* Tensor::Tensor(std::initializer_list<Type> values) */
template Tensor::Tensor(std::initializer_list<dtype_t<u32>> values);
template Tensor::Tensor(std::initializer_list<dtype_t<u64>> values);
template Tensor::Tensor(std::initializer_list<dtype_t<i32>> values);
template Tensor::Tensor(std::initializer_list<dtype_t<i64>> values);
template Tensor::Tensor(std::initializer_list<dtype_t<f32>> values);
template Tensor::Tensor(std::initializer_list<dtype_t<f64>> values);

/* Tensor::fill */
template void Tensor::fill<dtype_t<u32>>(dtype_t<u32> value, size_t count);
template void Tensor::fill<dtype_t<u64>>(dtype_t<u64> value, size_t count);
template void Tensor::fill<dtype_t<i32>>(dtype_t<i32> value, size_t count);
template void Tensor::fill<dtype_t<i64>>(dtype_t<i64> value, size_t count);
template void Tensor::fill<dtype_t<f32>>(dtype_t<f32> value, size_t count);
template void Tensor::fill<dtype_t<f64>>(dtype_t<f64> value, size_t count);

/* Tensor::item */
template dtype_t<u32> Tensor::item<u32>();
template dtype_t<u64> Tensor::item<u64>();
template dtype_t<i32> Tensor::item<i32>();
template dtype_t<i64> Tensor::item<i64>();
template dtype_t<f32> Tensor::item<f32>();
template dtype_t<f64> Tensor::item<f64>();

}