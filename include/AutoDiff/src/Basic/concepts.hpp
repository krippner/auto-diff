// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file concepts.hpp
 * @brief Defines concepts related to basic types.
 */

#ifndef AUTODIFF_SRC_BASIC_CONCEPTS_HPP
#define AUTODIFF_SRC_BASIC_CONCEPTS_HPP

#include "../Core/Expression.hpp" // ValueType

#include <concepts> // integral floating_point

namespace AutoDiff::Basic {

template <typename T>
concept Scalar = std::integral<T> || std::floating_point<T>;

template <typename Expr>
concept ExpressionType = requires { requires Scalar<ValueType_t<Expr>>; };

} // namespace AutoDiff::Basic

#endif // AUTODIFF_SRC_BASIC_CONCEPTS_HPP
