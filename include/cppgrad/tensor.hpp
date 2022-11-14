#ifndef CPPGRAD_TENSOR_HPP
#define CPPGRAD_TENSOR_HPP

#include <memory> // std::shared_ptr
#include <stdexcept> // std::runtime_error
#include <typeindex> // std::type_index
#include <vector> // std::vector

#include <numeric>

#include "cppgrad/config.hpp" // RTTI define
#include "cppgrad/device/device.hpp"

namespace cppgrad {

namespace impl {

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

}

class Tensor {
public:
    using DefaultType = float;

    template <typename T = DefaultType>
    static Tensor create(std::vector<size_t> shape = {},
        T fill_value = T { 0 },
        size_t alignment = alignof(T),
        Device* device = Device::get("cpu"))
    {
        size_t total_elements = std::reduce(shape.begin(), shape.end());
        auto strides = impl::make_strides(shape, sizeof(T));

        std::align_val_t align = alignment;
        device->allocate(total_elements * sizeof(T), align);

        return Tensor(_chunk, std::move(shape), std::move(strides), align, device);
    }

    bool empty() const noexcept
    {
        return _shape.size() == 0;
    }

    template <typename T = DefaultType>
    T item()
    {
        if (empty()) {
            throw std::range_error("Tensor is empty.");
        }

        if (_shape.size() > 1 || _shape[0] > 1) {
            throw std::range_error("Can only convert tensor of size 1 to a scalar.");
        }

#ifdef CPPGRAD_HAS_RTTI
        if (typeid(T) != _type_holder) {
            throw std::runtime_error("Requested type doesn't match content's type.");
        }
#endif
        // use device instead
        return *reinterpret_cast<T*>(_chunk);
    }

    // this may throw at std::shared_ptr(this) anyway
    Tensor operator[](size_t index)
    {
        if (empty()) {
            throw std::runtime_error("Trying to access empty Tensor.");
        }

        std::vector<size_t> new_shape { _shape.begin() + 1, _shape.end() };
        std::vector<size_t> new_strides { _strides.begin() + 1, _strides.end() };

        std::byte* new_chunk = _chunk + index * _strides[0];

        Tensor result { new_chunk, std::move(new_shape), std::move(new_strides), _alignment, _device };
        result._base = std::shared_ptr<Tensor>(this); // shared_ptr(this) but other way

        return result;
    }

    ~Tensor()
    {
        // check if _base is empty (says that it has no parent), and has allocated _chunk
        if (!_base && _chunk != nullptr) {
            _device->deallocate(_chunk, _alignment);
        }
    }

private:
    /*
            private constructor for indexing
    */
    Tensor(std::byte* chunk,
        std::vector<size_t>&& shape,
        std::vector<size_t>&& strides,
        std::align_val_t alignment,
        Device* device)
        : _chunk(chunk)
        , _shape(std::move(shape))
        , _strides(std::move(strides))
        , _alignment(alignment)
#ifdef CPPGRAD_HAS_RTTI
        , _type_holder(typeid(DefaultType))
#endif
    {
    }

    std::byte* _chunk { nullptr }; // std::byte* cuz it can possibly located on GPU
    std::align_val_t _alignment { 0 }; // we need to store that in order to successfully deallocate

    std::vector<size_t> _shape;
    std::vector<size_t> _strides;

    std::shared_ptr<Tensor> _base; // for views - holds pointer to original chunk holder

    Device* _device { nullptr };

#ifdef CPPGRAD_HAS_RTTI
    std::type_index _type_holder;
#endif

    /*
            we do need to make following things:
                    per-dimension strides (basically a step for which we should advance for chosen dimension)
                    stride-based indexing

            ... PROFIT!


    */
};

}

#endif