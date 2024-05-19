// Copyright (c) 2024 Matthias Krippner
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
#include "traits.hpp"             // isBasicType

// mandatory specializations of type traits for basic types

namespace AutoDiff::internal {

// Basic types are already equal to their evaluated types.
template <typename T>
struct Evaluated<T, std::enable_if_t<Basic::isBasicType_v<T>>> {
    using type = T;
};

// By default, only float values are paired with float derivatives...
template <>
struct DefaultDerivative<float> {
    using type = float;
};

// ...otherwise, use double derivatives.
template <typename T>
struct DefaultDerivative<T,
    std::enable_if_t<Basic::isBasicType_v<T> && !std::is_same_v<T, float>>> {
    using type = double;
};

} // namespace AutoDiff::internal

// implementation of type-specific operations

namespace AutoDiff::internal {

template <typename T>
struct TypeImpl<T, std::enable_if_t<Basic::isBasicType_v<T>>> {
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
using Integer = Variable<int, double>;
using Boolean = Variable<bool, double>;

using RealF    = Variable<float, float>;
using IntegerF = Variable<int, float>;
using BooleanF = Variable<bool, float>;

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_MODULE_HPP
