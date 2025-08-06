// Copyright (c) 2024-2025 Matthias Krippner
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

#include "concepts.hpp"

#define AUTODIFF_MAKE_BASIC_UNARY_OP(operation, Type)                          \
    template <Basic::ExpressionType X>                                         \
    auto operation(Expression<X> const& x)                                     \
    {                                                                          \
        return Type<X>(x);                                                     \
    }

#define AUTODIFF_MAKE_BASIC_BINARY_OP(operation, Type)                         \
    template <Basic::ExpressionType X, Basic::ExpressionType Y>                \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <Basic::Scalar X, Basic::ExpressionType Y>                        \
    auto operation(X x, Expression<Y> const& y)                                \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <Basic::ExpressionType X, Basic::Scalar Y>                        \
    auto operation(Expression<X> const& x, Y y)                                \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }

#endif // AUTODIFF_SRC_BASIC_FACTORIES_HPP
