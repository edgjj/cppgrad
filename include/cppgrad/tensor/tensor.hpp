// Copyright (c) 2023 Yegor Suslin
// 
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CPPGRAD_TENSOR_HPP
#define CPPGRAD_TENSOR_HPP

#include <memory> // std::shared_ptr
#include <type_traits> // std::enable_if
#include <vector> // std::vector

#include <any> // std::any
#include <numeric> // std::reduce

#include "cppgrad/tensor/impl.hpp"
#include "cppgrad/tensor/util/impl_util.hpp"

// exceptions
#include "cppgrad/exceptions/index_error.hpp"

// ops
#include "cppgrad/device/executor.hpp"
#include "cppgrad/tensor/ops/op_overloads.hpp"

// autograd
#include "cppgrad/autograd/context.hpp"
#include "cppgrad/autograd/grad_mode.hpp"

// comm
#include "cppgrad/distributed/comm.hpp"

namespace cppgrad {

namespace impl {
    struct Executor;
}

/*
    may need a .contiguous() type thing to make transposed matrices flat
*/
class Tensor : public autograd::AutogradInterface {
public:
    /**
     * @brief Returns a Tensor of chosen shape and device, filled with some value.
     *
     * @tparam T
     * @param shape
     * @param fill_value
     * @param device
     * @return Tensor
     */

    template <typename T = float>
    static Tensor full(const std::vector<size_t>& shape = {},
        T fill_value = T { 0 },
        DType data_type = f32,
        std::any device_tag = kCPU)
    {
        auto result = create_dirty(shape, data_type, device_tag);
        result.fill(fill_value);

        return result;
    }

    template <typename T = float>
    static Tensor rand(const std::vector<size_t>& shape,
        T lower_bound = T { -1 },
        T upper_bound = T { 1 },
        DType data_type = f32,
        std::any device_tag = kCPU)
    {
        auto result = create_dirty(shape, data_type, device_tag);
        result.random_fill(lower_bound, upper_bound);

        return result;
    }

    static Tensor zeros(const std::vector<size_t>& shape,
        DType data_type = f32,
        std::any device_tag = kCPU)
    {
        return full(shape, 0, data_type, std::move(device_tag));
    }

    static Tensor ones(const std::vector<size_t>& shape,
        DType data_type = f32,
        std::any device_tag = kCPU)
    {
        return full(shape, 1, data_type, std::move(device_tag));
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
    template <DType DataType> // we keep DType here since we cant cast blobs
    static Tensor from_blob(dtype_t<DataType>* blob,
        const std::vector<size_t>& shape = {},
        std::any device_tag = kCPU)
    {
        auto result = create_dirty(shape, DataType, device_tag);
        constexpr size_t type_sz = sizeof(dtype_t<DataType>);

        /**
         *  We use HostToDevice there since DeviceType might be CUDA, or other non-CPU type.
         *  On CPU it's no-op.
         */
        result.executor().copy(reinterpret_cast<std::byte*>(blob), result.data(), result.nbytes(), impl::HostToDevice);
        return result;
    }

    /**
     * @brief Indexing operator. Returns tensor wrapping a view onto current tensor data.
     *
     * @param index
     * @return Tensor
     */

    template <typename... Indices>
    Tensor operator()(Indices... variadic_index) const
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
            _storage->_device,
            _storage->_type_id };

        result._base = base_storage();

        return result;
    }

    /**
     *
     * @brief Returns a Tensor of chosen shape, alignment and device, containing unitialized memory.
     *
     * Should be used for custom initialization routines.
     *
     * @param shape
     * @param type
     * @param alignment
     * @param device
     * @return Tensor
     */
    static Tensor create_dirty(const std::vector<size_t>& shape,
        DType type,
        std::any device_tag);

    /**
     * @brief Creates dirty Tensor with specs of another Tensor
     *
     * @param other
     * @return Tensor
     */
    static Tensor create_dirty(const Tensor& other);

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
    Tensor& operator=(const Tensor& other);

    Tensor& operator=(Tensor&& other);

    // this ensures that _storage won't be empty
    Tensor();

    template <typename Type, std::enable_if_t<std::is_arithmetic_v<Type>>* = nullptr>
    Tensor(Type value, DType dtype = rtype_v<Type>)
    {
        *this = full({ 1 }, value, dtype);
    }

    template <typename Type>
    Tensor(std::initializer_list<Type> values, DType dtype = rtype_v<Type>)
    {
        for_each_type(
            [&, values](auto tag) {
                using CastedType = decltype(tag);

                std::vector<CastedType> imr_vector;

                for (auto& v : values) {
                    imr_vector.push_back(v);
                }

                *this = from_blob<rtype_v<CastedType>>(imr_vector.data(), { imr_vector.size() });
            },
            dtype);
    }

    Tensor(std::initializer_list<Tensor> values);

    /**
     * @brief Tensor equality comparison
     * Doesn't compare Tensors actual contents, compares Tensor's storages.
     *
     * @param rhs
     * @return true
     * @return false
     */
    bool operator==(const Tensor& rhs) const;

    /**
     * @brief Fills Tensor, autocasted
     *
     * @tparam T
     * @param value
     */
    template <typename Type>
    void fill(Type value)
    {
        for_each_type(
            [&, value](auto tag) {
                using CastedType = decltype(tag);
                auto casted_value = CastedType(value);
                auto* byte_ptr = reinterpret_cast<std::byte*>(&casted_value);

                executor().fill(*this, byte_ptr);
            },
            dtype());
    }

    template <typename Type = double>
    void random_fill(Type lower_bound = -1.0, Type upper_bound = 1.0)
    {
        // must do the same casting thing as fill()
        executor().random_fill(*this, lower_bound, upper_bound);
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
    dtype_t<DataType> item() const;

    /**
     * @brief Indexing operator. 1-dim operator() wrapper.
     *
     * @param index
     * @return Tensor
     */
    Tensor operator[](size_t index) const;

    /**
     * @brief Retrieves current data type stored in Tensor.
     *
     * @return DType type index
     */
    DType dtype() const;

    /**
     * @brief Gets raw pointer to Tensor data.
     *
     * You must only use it if there's a REALLY good reason to do so.
     *
     * @return std::byte* raw data pointer
     */
    std::byte* data();

    /**
     * @brief Const version of data()
     *
     * @return const std::byte*
     */
    const std::byte* data() const;

    /**
     * @brief Tells whether tensor is empty or not.
     *
     * @return bool
     */
    bool empty() const noexcept;

    /**
     * @brief Return total count of elements stored in Tensor
     *
     * @return size_t
     */
    size_t numel() const noexcept;

    /**
     * @brief Return size of data chunk in bytes
     *
     * @return size_t
     */
    size_t nbytes() const noexcept;

    /**
     * @brief Return current tensor shape.
     *
     * @return const std::vector<size_t>&
     */
    const std::vector<size_t>& shape() const noexcept;

    /**
     * @brief Returns data indexing strides.
     *
     * @return const std::vector<size_t>&
     */
    const std::vector<size_t>& strides() const noexcept;

    /**
     * @brief Returns parent "base" tensor, in case if current tensor is view onto parent data.
     *        Otherwise returns shallow copy of current tensor.
     *
     * @return Tensor
     */
    Tensor base();

    /**
     * @brief Makes transposed view to Tensor.
     *
     * @return Tensor transposed Tensor
     */
    Tensor T() const;

    /**
     * @brief Return tag representing current device.
     * @return DeviceTag
     */
    DeviceTag device() const;

    /**
     * @brief Utility method to get executor of attached device
     * UB if Tensor has no device.
     *
     * @return impl::Executor&
     */
    impl::Executor& executor() const;

    /**
     * @brief Checks if current Tensor data is located on CUDA device.
     *
     */
    bool is_cuda_tensor() const;

    /**
     * @brief Returns a new Tensor, having same data, but located on CUDA device.
     * Note: this is not-differentiable, wanishes grad history.
     *
     * @return Tensor
     */
    Tensor cuda() const;

    /**
     * @brief Returns a new Tensor, having same data, but located on CPU memory.
     * Note: this is not-differentiable, wanishes grad history.
     *
     * @return Tensor
     */
    Tensor cpu() const;

    /**
     * @brief Clones a Tensor.
     * Note: this is not-differentiable, wanishes grad history.
     *
     * @return Tensor
     */
    Tensor clone() const;

    /**
     * @brief Makes new Tensor from current, with data stored contiguous
     * Note: this is not-differentiable, wanishes grad history.
     *
     * @return Tensor
     */
    Tensor contiguous() const;

    /**
     * @brief Returns shallow copy of Tensor, which fakes shape, and has infinite length.
     * Basically, a Tensor with 0 stride.
     *
     * Only works with single-element Tensors.
     *
     * @return Tensor
     */
    Tensor loop(const std::vector<size_t>& fake_shape) const;

    /**
     * @brief Determines if Tensor is view onto other Tensor
     *
     */
    bool is_view() const;

    /**
     * @brief Determines if Tensor's data stored contiguous.
     * Might be useful for working with transposed Tensors.
     *
     */
    bool is_contiguous() const;

    /**
     * @brief Tells is Tensor is a loop
     *
     * @return true
     * @return false
     */
    bool is_loop() const;

    /*
        @brief Autograd methods go here;

        For documentation @see AutogradInterface
    */
    Tensor& grad() override;
    const Tensor& grad() const override;
    void set_grad_fn(std::shared_ptr<autograd::Node> new_grad_fn) override;
    std::shared_ptr<autograd::Node>& grad_fn() override;
    void set_requires_grad(bool new_requires_grad) override;
    bool requires_grad() const override;

    /**
     * @brief Runs backward pass starting from this Tensor.
     *
     */
    void backward();

    ~Tensor();

    friend struct std::hash<Tensor>;

    // these two allow to set strides
    friend struct PermuteOp;
#ifdef CPPGRAD_HAS_MPI
    friend class distributed::Communicator;
#endif

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
        const std::vector<size_t>& shape,
        const std::vector<size_t>& strides,
        DeviceTag device,
        DType type);

    /**
     * @brief Utility method to get actual base TensorData storage pointer.
     *
     * @return std::shared_ptr<impl::TensorData>& Parent's TensorData
     */
    std::shared_ptr<impl::TensorData>& base_storage();

    const std::shared_ptr<impl::TensorData>& base_storage() const;

    static void checked_copy(const Tensor& from, Tensor& to);

    /**
     * @brief Construct a new Tensor object from parent's TensorData.
     *
     * @param base_storage Parent's TensorData
     */
    Tensor(std::shared_ptr<impl::TensorData> base_storage);

private:
    std::shared_ptr<impl::TensorData> _storage;
    // duplicate for views
    std::shared_ptr<impl::TensorData> _base;
};

}

// hash specialization
template <>
struct std::hash<cppgrad::Tensor> {
    [[nodiscard]] size_t operator()(const cppgrad::Tensor& tensor) const noexcept
    {
        return std::hash<std::shared_ptr<cppgrad::impl::TensorData>>()(tensor._storage);
    }
};

#endif