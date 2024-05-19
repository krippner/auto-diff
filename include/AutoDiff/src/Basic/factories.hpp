// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file factories.hpp
 * @brief Macros defining factory functions for basic operations.
 *
 * Use these macros inside the @c AutoDiff namespace. They can be used without
 * namespace qualification in user code because of argument-dependent lookup.
 */

#ifndef AUTODIFF_SRC_BASIC_FACTORIES_HPP
#define AUTODIFF_SRC_BASIC_FACTORIES_HPP

#include "traits.hpp"

#define AUTODIFF_MAKE_BASIC_UNARY_OP(operation, Type)                          \
    template <typename X>                                                      \
    auto operation(Expression<X> const& x)                                     \
        -> std::enable_if_t<Basic::hasBasicValue_v<X>, Type<X>>                \
    {                                                                          \
        return Type<X>(x);                                                     \
    }

#define AUTODIFF_MAKE_BASIC_BINARY_OP(operation, Type)                         \
    template <typename X, typename Y>                                          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
        -> std::enable_if_t<Basic::hasBasicValue_v<X>                          \
                                && Basic::hasBasicValue_v<Y>,                  \
            Type<X, Y>>                                                        \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Scalar, typename Y>                                     \
    auto operation(Scalar x, Expression<Y> const& y)                           \
        -> std::enable_if_t<Basic::isBasicType_v<Scalar>                       \
                                && Basic::hasBasicValue_v<Y>,                  \
            Type<Scalar, Y>>                                                   \
    {                                                                          \
        return Type<Scalar, Y>(x, y);                                          \
    }                                                                          \
                                                                               \
    template <typename X, typename Scalar>                                     \
    auto operation(Expression<X> const& x,                                     \
        Scalar y) -> std::enable_if_t<Basic::hasBasicValue_v<X>                \
                                          && Basic::isBasicType_v<Scalar>,     \
                      Type<X, Scalar>>                                         \
    {                                                                          \
        return Type<X, Scalar>(x, y);                                          \
    }

#endif // AUTODIFF_SRC_BASIC_FACTORIES_HPP
