// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file traits.hpp
 * @brief Type traits for expression templates.
 *
 * These traits are used by the @c Variable and @c internal::Evaluator classes
 * to determine the value and derivative types of expression templates.
 * They must be specialized for new expression templates.
 */

#ifndef AUTODIFF_SRC_INTERNAL_TRAITS_HPP
#define AUTODIFF_SRC_INTERNAL_TRAITS_HPP

namespace AutoDiff::internal {

/**
 * @brief Type trait mapping expression templates to evaluated types.
 *
 * For example, the expression template @c Sum<Real,Real> might evaluate
 * to @c Real.
 *
 * @note Specialize this template for supported expression templates.
 */
template <typename T, typename = void>
struct Evaluated;

/**
 * @brief The type an expression template evaluates to.
 *
 * Depends on module implementation.
 */
template <typename T>
using Evaluated_t = typename Evaluated<T>::type;

/**
 * @brief Type trait mapping value types to default derivative types.
 *
 * Mathematically, the value type represents points on a manifold and the
 * derivative type represents tangent vectors to that manifold.
 * Hence, the choice of tangent space depends on the choice of manifold.
 *
 * @note Specialize this template for supported value types.
 */
template <typename T, typename = void>
struct DefaultDerivative;

/**
 * @brief The default derivative type associated with a value type.
 *
 * Depends on module implementation.
 */
template <typename T>
using DefaultDerivative_t = typename DefaultDerivative<T>::type;

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_TRAITS_HPP
