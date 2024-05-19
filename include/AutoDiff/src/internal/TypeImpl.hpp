// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_INTERNAL_TYPE_IMPL_HPP
#define AUTODIFF_SRC_INTERNAL_TYPE_IMPL_HPP

#include "Shape.hpp" // Shape MapDescription

namespace AutoDiff::internal {

/**
 * @brief Implementation of type-specific operations.
 *
 * These operations are used by the @c Computation and @c Evaluator class
 * to perform certain operations on values and derivatives.
 * See internal/TypeImpl.hpp to see which operations are required.
 *
 * @note Specialize this template for supported value and derivative types.
 */
template <typename T, typename = void>
struct TypeImpl;

/**
 * @brief Returns the shape of a value.
 *
 * The shape of a value is the shape of the array representing that value.
 */
template <typename T>
auto getShape(const T& value) -> Shape
{
    return TypeImpl<T>::getShape(value);
}

/**
 * @brief Returns the shape of the codomain of a derivative.
 */
template <typename T>
auto codomainShape(const T& derivative) -> Shape
{
    return TypeImpl<T>::codomainShape(derivative);
}

/**
 * @brief Generates a derivative from description.
 *
 * In particular, this function generates zero and identity maps.
 */
template <typename T>
void generate(T& derivative, MapDescription const& descr)
{
    TypeImpl<T>::generate(derivative, descr);
}

/**
 * @brief Assign other to value.
 */
template <typename T, typename Other>
void assign(T& value, Other const& other)
{
    TypeImpl<T>::assign(value, other);
}

/**
 * @brief Add-assign other to value.
 */
template <typename T, typename Other>
void addTo(T& value, Other const& other)
{
    TypeImpl<T>::addTo(value, other);
}

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_TYPE_IMPL_HPP
