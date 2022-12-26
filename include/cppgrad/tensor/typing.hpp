#ifndef CPPGRAD_TENSOR_TYPES_HPP
#define CPPGRAD_TENSOR_TYPES_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace cppgrad {

enum DType : uint64_t {
    u32,
    u64,
    i32,
    i64,
    f32,
    f64,
    undefined = 0xFFFF
};

namespace impl {

    using TypesTuple = std::tuple<uint32_t, uint64_t, int32_t, int64_t, float, double>;

    template <DType T>
    struct Type {
        // will probably cause weird error message on invalid types but we're with that
        using type = std::tuple_element_t<T, TypesTuple>;
    };

    // thanks dear comment pal: https://devblogs.microsoft.com/oldnewthing/20200629-00/?p=103910
    template <size_t index, typename T, typename Tuple>
    constexpr size_t tuple_element_index_helper()
    {
        if constexpr (index == std::tuple_size_v<Tuple>) {
            return index;
        } else {
            return std::is_same_v<T, std::tuple_element_t<index, Tuple>> ? index : tuple_element_index_helper<index + 1, T, Tuple>();
        }
    }

    template <typename T>
    constexpr auto tuple_element_index()
    {
        return tuple_element_index_helper<0, T, TypesTuple>();
    }

    template <size_t... I>
    constexpr auto get_sizes_helper(std::index_sequence<I...>)
    {
        constexpr std::array<size_t, sizeof...(I)> sizes = { sizeof(std::tuple_element_t<I, TypesTuple>)... };
        return sizes;
    }

    constexpr auto get_sizes()
    {
        return get_sizes_helper(std::make_index_sequence<std::tuple_size_v<TypesTuple>> {});
    }

    template <typename T, typename Fn, typename... Args>
    constexpr void for_each_type_matcher(Fn&& fun, DType type, Args&&... args)
    {
        if (type == (DType)tuple_element_index<T>()) {
            fun(T {});
        }
    }

    template <typename Fn, DType... types>
    constexpr void for_each_type_impl(Fn&& fun, DType type)
    {
        (for_each_type_matcher<typename impl::Type<types>::type>(std::forward<Fn>(fun), type), ...);
    }
}

template <DType T>
using dtype_t = typename impl::Type<T>::type;

template <typename T>
inline constexpr DType rtype_v = (DType)impl::tuple_element_index<T>();

constexpr size_t dtype_size(DType type)
{
    constexpr auto sizes = impl::get_sizes();
    return sizes[type];
}

constexpr const char* dtype_name(DType type)
{
    constexpr std::array<const char*, 7> names = { "u32", "u64", "i32", "i64", "f32", "f64", "undef" };
    return names[type];
}

template <typename Fn>
constexpr void for_each_type(Fn&& fun, DType type)
{
    // this looks much better
    impl::for_each_type_impl<Fn, u32, u64, i32, i64, f32, f64>(std::forward<Fn>(fun), type);
}

// non-lambda macro, takes things by ref atm
// should be ok on CUDA, as destructor/item() 'd synchronize
#define FOREACH_TYPE(type, fn, ...) \
    for_each_type([&](auto tag) { fn<decltype(tag)>(__VA_ARGS__); }, type);

}

#endif