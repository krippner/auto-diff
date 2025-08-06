// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file factories.hpp
 * @brief Macros defining factory functions for operations on Eigen arrays.
 *
 * Use these macros inside the @c AutoDiff namespace. They can be used without
 * namespace qualification in user code because of argument-dependent lookup.
 */

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_FACTORIES_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_FACTORIES_HPP

#include "../concepts.hpp"

#define AUTODIFF_MAKE_ARRAY_UNARY_OP(operation, Type)                          \
    template <EigenAD::ArrayExpression X>                                      \
    auto operation(Expression<X> const& x)                                     \
    {                                                                          \
        return Type<X>(x);                                                     \
    }

#define AUTODIFF_MAKE_ARRAY_ARRAY_OP(operation, Type)                          \
    template <EigenAD::ArrayExpression X, EigenAD::ArrayExpression Y>          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Derived, EigenAD::ArrayExpression Y>                    \
    auto operation(Eigen::ArrayBase<Derived> const& x, Expression<Y> const& y) \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <EigenAD::ArrayExpression X, typename Derived>                    \
    auto operation(Expression<X> const& x, Eigen::ArrayBase<Derived> const& y) \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#define AUTODIFF_MAKE_ARRAY_SCALAR_OP(operation, Type)                         \
    template <EigenAD::ArrayExpression X, EigenAD::Scalar Y>                   \
    auto operation(Expression<X> const& x, Y y)                                \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <EigenAD::ArrayExpression X, EigenAD::ScalarExpression Y>         \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }

#define AUTODIFF_MAKE_SCALAR_ARRAY_OP(operation, Type)                         \
    template <EigenAD::Scalar X, EigenAD::ArrayExpression Y>                   \
    auto operation(X x, Expression<Y> const& y)                                \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <EigenAD::ScalarExpression X, EigenAD::ArrayExpression Y>         \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }

#define AUTODIFF_MAKE_ARRAY_BINARY_OP(operation, Type)                         \
    AUTODIFF_MAKE_ARRAY_ARRAY_OP(operation, Type)                              \
    AUTODIFF_MAKE_ARRAY_SCALAR_OP(operation, Type)                             \
    AUTODIFF_MAKE_SCALAR_ARRAY_OP(operation, Type)

#endif // AUTODIFF_SRC_EIGEN_ARRAY_FACTORIES_HPP
