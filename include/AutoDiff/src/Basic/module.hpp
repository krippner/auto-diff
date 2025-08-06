// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file module.hpp
 * @brief Defines necessary type traits to enable AutoDiff for basic types.
 *
 * Additionally, this file provides aliases for common @c Variable types.
 */

#ifndef AUTODIFF_SRC_BASIC_MODULE_HPP
#define AUTODIFF_SRC_BASIC_MODULE_HPP

#define AUTODIFF_MODULE // only one module per translation unit

#include "../internal/TypeImpl.hpp"
#include "../internal/traits.hpp" // traits to be specialized
#include "concepts.hpp"           // Scalar

#include <type_traits> // conditional

// mandatory specializations of type traits for basic types

namespace AutoDiff::internal {

// Basic types are already equal to their evaluated types.
template <Basic::Scalar T>
struct Evaluated<T> {
    using type = T;
};

template <Basic::Scalar T>
struct DefaultDerivative<T> {
    using type = std::conditional_t<std::is_same_v<T, float>,
        float, // float value -> float derivative
        double // otherwise
        >;
};

} // namespace AutoDiff::internal

// implementation of type-specific operations

namespace AutoDiff::internal {

template <Basic::Scalar T>
struct TypeImpl<T> {
    static auto getShape(T const& /*value*/) -> Shape { return {1}; }

    static auto codomainShape(T const /*derivative*/) -> Shape { return {1}; }

    static void generate(T& derivative, MapDescription const& descr)
    {
        if (descr.state == MapDescription::zero) {
            derivative = static_cast<T>(0);
        } else if (descr.state == MapDescription::identity) {
            derivative = static_cast<T>(1);
        }
    }

    static void assign(T& value, T const& other) { value = other; }

    static void addTo(T& value, T const& other) { value += other; }
};

} // namespace AutoDiff::internal

// aliases

namespace AutoDiff {

template <typename Value, typename Derivative>
class Variable;

using Real    = Variable<double, double>;
using RealF   = Variable<float, float>;
using Integer = Variable<int, double>;
using Boolean = Variable<bool, double>;

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_MODULE_HPP
