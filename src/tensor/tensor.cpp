#include <algorithm> // std::is_sorted

#include "cppgrad/exceptions/generic_error.hpp"
#include "cppgrad/exceptions/type_error.hpp"
#include "cppgrad/tensor/tensor.hpp"

#include "cppgrad/device/registry.hpp"
#include "cppgrad/device/tags.hpp"

#include "cppgrad/tensor/ops/grad_ops.hpp"

namespace cppgrad {

// util routine for inter-device copies & various clones
void Tensor::checked_copy(const Tensor& from, Tensor& to)
{
    // set strides to match, can't do strided copy there
    to._storage->_strides = from._storage->_strides;

    size_t n_bytes = from.nbytes();

    if (from.device() == to.device()) { // homogeneous
        from.executor().strided_copy(from, to); // faster than memcpy
    } else if (from.is_cuda_tensor()) {
        from.executor().copy(from.data(), to.data(), n_bytes, impl::DeviceToHost);
    } else {
        to.executor().copy(from.data(), to.data(), n_bytes, impl::HostToDevice);
    }
}

Tensor Tensor::create_dirty(const std::vector<size_t>& shape,
    DType type,
    std::any device_tag)
{
    auto type_size = dtype_size(type);
    size_t total_elements = std::reduce(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    auto strides = impl::make_strides(shape, type_size);

    auto device_numeric_tag = DeviceRegistry::get()
        .get_numeric_tag(device_tag);

    auto* chunk = DeviceRegistry::get()
        .by_numeric(device_numeric_tag)
        .allocate(total_elements * type_size);

    return Tensor(chunk,
        shape,
        strides,
        device_numeric_tag,
        type);
}

Tensor Tensor::create_dirty(const Tensor& other)
{
    return create_dirty(other.shape(), other.dtype(), other.device());
}

Tensor& Tensor::operator=(const Tensor& other)
{
    // just return shallow copy for empty case / not views
    if (empty() || !is_view() && !other.is_view() || is_loop() && other.is_loop()) {
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

    CPPGRAD_CHECK_EQ(device(), other.device(),
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
    // just move if empty or both are not views
    if (empty() || !is_view() && !other.is_view() || is_loop() && other.is_loop()) {
        _storage = std::move(other._storage);
        _base = std::move(other._base);
        return *this;
    }

    // check if assignment requirements satisfy
    CPPGRAD_CHECK_EQ(shape(), other.shape(),
        exceptions::GenericError,
        "Assign (move) shape mismatch");

    CPPGRAD_CHECK_EQ(dtype(), other.dtype(),
        exceptions::GenericError,
        "Assign (move) DType mismatch");

    CPPGRAD_CHECK_EQ(device(), other.device(),
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
{
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

    *this = create_dirty(shape, base_dtype, kCPU); // copy device ?

    for (size_t i = 0; i < values.size(); i++) {
        (*this)(i) = *(values.begin() + i);
    }
}

bool Tensor::operator==(const Tensor& rhs) const
{
    return _storage == rhs._storage;
}

template <DType DataType>
dtype_t<DataType> Tensor::item() const
{
    CPPGRAD_CHECK_FALSE(empty(),
        exceptions::IndexError);

    CPPGRAD_CHECK_FALSE(numel() > 1,
        exceptions::GenericError,
        "Can only convert tensor having 1 element to a scalar.");

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

Tensor Tensor::operator[](size_t index) const
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
    if (is_loop()) {
        return dtype_size(dtype());
    }

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

// ok to const; has nothing to deal with original Tensor
Tensor Tensor::T() const
{
    std::vector<size_t> perm(shape().size());
    std::iota(perm.rbegin(), perm.rend(), 0);

    return PermuteOp::apply({ *this }, std::move(perm))[0];
}

DeviceTag Tensor::device() const
{
    return base_storage()->_device;
}

#ifdef CPPGRAD_HAS_CUDA
bool Tensor::is_cuda_tensor() const
{
    // we assume that there's always device attached to Tensor
    return device() == kCUDA;
}

Tensor Tensor::cuda() const
{
    // its ok if no cuda devices -> cuda allocator will throw
    if (is_cuda_tensor()) {
        return *this;
    }

    auto new_tensor = create_dirty(shape(), dtype(), kCUDA);
    checked_copy(*this, new_tensor);

    if (requires_grad()) {
        new_tensor.set_requires_grad(true);
    }

    return new_tensor;
}

Tensor Tensor::cpu() const
{
    // no need in cast
    if (!is_cuda_tensor()) {
        return *this;
    }

    auto new_tensor = create_dirty(shape(), dtype(), kCPU);
    checked_copy(*this, new_tensor);

    if (requires_grad()) {
        new_tensor.set_requires_grad(true);
    }

    return new_tensor;
}

#else
// let it be just no-op if CUDA capatibilities are not built
bool Tensor::is_cuda_tensor() const
{
    return false;
}

Tensor Tensor::cpu() const
{
    return *this;
}

Tensor Tensor::cuda() const
{
    return *this;
}
#endif

Tensor Tensor::clone() const
{
    auto new_tensor = create_dirty(shape(), dtype(), device());
    checked_copy(*this, new_tensor);
    return new_tensor;
}

Tensor Tensor::contiguous() const
{
    // just return copy
    if (is_contiguous()) {
        return *this;
    }

    auto new_tensor = create_dirty(shape(), dtype(), device());
    executor().strided_copy(*this, new_tensor);

    return new_tensor;
}

Tensor Tensor::loop(const std::vector<size_t>& fake_shape) const
{
    // check if 1-dim and 1 element
    if (shape().size() != 1 || shape()[0] != 1) {
        throw exceptions::GenericError("Looping requires Tensor have 1 element available");
    }

    // fake shape and zero out stride
    std::vector<size_t> new_shape { fake_shape };
    std::vector<size_t> new_strides { 0 };

    Tensor result { _storage->_chunk,
        new_shape,
        new_strides,
        _storage->_device,
        _storage->_type_id };

    result._base = base_storage();

    return result;
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

bool Tensor::is_loop() const
{
    return !empty() && _storage->_strides.size() == 1 && _storage->_strides[0] == 0;
}

Tensor& Tensor::grad()
{
    if (!_storage->_autograd_context) {
        _storage->_autograd_context = autograd::impl::AutogradContextFactory::make();
    }

    return _storage->_autograd_context->grad();
};

const Tensor& Tensor::grad() const
{
    if (!_storage->_autograd_context) {
        return autograd::impl::AutogradContextFactory::empty_tensor();
    }

    return _storage->_autograd_context->grad();
}

void Tensor::set_grad_fn(std::shared_ptr<autograd::Node> new_grad_fn)
{
    if (!_storage->_autograd_context) {
        _storage->_autograd_context = autograd::impl::AutogradContextFactory::make();
    }

    _storage->_autograd_context->set_grad_fn(std::move(new_grad_fn));
}

std::shared_ptr<autograd::Node>& Tensor::grad_fn()
{
    if (!_storage->_autograd_context) {
        _storage->_autograd_context = autograd::impl::AutogradContextFactory::make();
    }

    return _storage->_autograd_context->grad_fn();
}

void Tensor::set_requires_grad(bool new_requires_grad)
{
    if (!_storage->_autograd_context) {
        _storage->_autograd_context = autograd::impl::AutogradContextFactory::make();
    }

    _storage->_autograd_context->set_requires_grad(new_requires_grad);
}

bool Tensor::requires_grad() const
{
    if (!_storage->_autograd_context) {
        return false;
    }

    return _storage->_autograd_context->requires_grad();
}

void Tensor::backward()
{
    autograd::backward(*this);
}

Tensor::~Tensor()
{
    // check if we have parent TensorData attached
    auto& actual_storage = base_storage();

    if (actual_storage.use_count() == 1) {
        DeviceRegistry::get().by_numeric(device()).deallocate(actual_storage->_chunk);
    }
}

Tensor::Tensor(std::byte* chunk,
    const std::vector<size_t>& shape,
    const std::vector<size_t>& strides,
    DeviceTag device,
    DType type)
{
    _storage = std::make_shared<impl::TensorData>(
        chunk,
        shape,
        strides,
        device,
        type);

    // init if forced grad
    if (autograd::ThreadLocalGradState::get()) {
        _storage->_autograd_context = autograd::impl::AutogradContextFactory::make();
    }
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
    return DeviceRegistry::get().by_numeric(device()).get_executor();
}

Tensor::Tensor(std::shared_ptr<impl::TensorData> base_storage)
    : _storage(std::move(base_storage))
{
}

// template instantiations go here

/* Tensor::item */
template dtype_t<u32> Tensor::item<u32>() const;
template dtype_t<u64> Tensor::item<u64>() const;
template dtype_t<i32> Tensor::item<i32>() const;
template dtype_t<i64> Tensor::item<i64>() const;
template dtype_t<f32> Tensor::item<f32>() const;
template dtype_t<f64> Tensor::item<f64>() const;

}