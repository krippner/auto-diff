// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file traits.hpp
 * @brief Defines which types are considered basic types.
 */

#ifndef AUTODIFF_SRC_BASIC_TRAITS_HPP
#define AUTODIFF_SRC_BASIC_TRAITS_HPP

#include "../Core/Expression.hpp" // ValueType

#include <type_traits> // is_arithmetic

namespace AutoDiff::Basic {

namespace detail {

    template <typename T>
    using IsBasicType = std::is_arithmetic<T>;

    template <typename Expr>
    using HasBasicValue = IsBasicType<ValueType_t<Expr>>;

} // namespace detail

template <typename T>
constexpr bool isBasicType_v
    = detail::IsBasicType<T>::value; // NOLINT(*-type-traits)

template <typename Expr>
constexpr bool hasBasicValue_v
    = detail::HasBasicValue<Expr>::value; // NOLINT(*-type-traits)

} // namespace AutoDiff::Basic

#endif // AUTODIFF_SRC_BASIC_TRAITS_HPP
