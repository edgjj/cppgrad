#ifndef CPPGRAD_ITERTOOLS_COMBINE_HPP
#define CPPGRAD_ITERTOOLS_COMBINE_HPP
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cppgrad::itertools {

template <typename... Iterators>
struct combined_iterator {
    using IterTuple = std::tuple<Iterators...>;

    using reference = std::tuple<typename std::iterator_traits<Iterators>::reference...>;

    combined_iterator(IterTuple it)
        : _iter_holder(it)
    {
    }

    bool operator==(combined_iterator<Iterators...> rhs) const
    {
        return std::get<0>(_iter_holder) == std::get<0>(rhs._iter_holder);
    }

    bool operator!=(combined_iterator<Iterators...> rhs) const
    {
        return std::get<0>(_iter_holder) != std::get<0>(rhs._iter_holder);
    }

    void operator++()
    {
        std::apply([](auto&&... v) { (v++, ...); }, _iter_holder);
    }

    reference operator*()
    {
        return expand(std::make_index_sequence<sizeof...(Iterators)> {});
    }

private:
    template <std::size_t... I>
    reference expand(std::index_sequence<I...>)
    {
        return reference(*std::get<I>(_iter_holder)...);
    }

    IterTuple _iter_holder;
};

template <typename... Iterators>
struct combined_range {
    using IterTuple = std::tuple<Iterators...>;

    combined_range(IterTuple begin, IterTuple end)
        : _begin(begin)
        , _end(end)
    {
    }

    auto begin()
    {
        return _begin;
    }

    auto end()
    {
        return _end;
    }

private:
    combined_iterator<Iterators...> _begin;
    combined_iterator<Iterators...> _end;
};

template <typename... Ranges>
auto combine(Ranges&&... ranges)
{
    return combined_range(std::make_tuple(std::begin(std::forward<Ranges>(ranges))...),
        std::make_tuple(std::end(std::forward<Ranges>(ranges))...));
}

}

#endif