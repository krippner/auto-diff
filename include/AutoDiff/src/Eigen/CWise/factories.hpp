// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file factories.hpp
 * @brief Macros defining factory functions for elementwise operations
 * on Eigen matrices.
 *
 * Use these macros inside the @c AutoDiff namespace. They can be used without
 * namespace qualification in user code because of argument-dependent lookup.
 */

#ifndef AUTODIFF_SRC_EIGEN_CWISE_FACTORIES_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_FACTORIES_HPP

#include "../concepts.hpp"

#define AUTODIFF_MAKE_CWISE_UNARY_OP(operation, Type)                          \
    template <EigenAD::MatrixBaseExpression X>                                 \
    auto operation(Expression<X> const& x)                                     \
    {                                                                          \
        return Type<X>(x);                                                     \
    }

#define AUTODIFF_MAKE_CWISE_BINARY_OP(operation, Type)                         \
    template <EigenAD::MatrixBaseExpression X,                                 \
        EigenAD::MatrixBaseExpression Y>                                       \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Derived, EigenAD::MatrixBaseExpression Y>               \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <EigenAD::MatrixBaseExpression X, typename Derived>               \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#define AUTODIFF_MAKE_CWISE_SCALAR_OP(operation, Type)                         \
    template <EigenAD::MatrixBaseExpression X, EigenAD::ScalarExpression Y>    \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Derived, EigenAD::ScalarExpression Y>                   \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <EigenAD::MatrixBaseExpression X, EigenAD::Scalar Y>              \
    auto operation(Expression<X> const& x, Y y)                                \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }

#define AUTODIFF_MAKE_CWISE_SCALAR_MATRIX_OP(operation, Type)                  \
    template <EigenAD::ScalarExpression X, EigenAD::MatrixBaseExpression Y>    \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <EigenAD::ScalarExpression X, typename Derived>                   \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }                                                                          \
                                                                               \
    template <EigenAD::Scalar X, EigenAD::MatrixBaseExpression Y>              \
    auto operation(X x, Expression<Y> const& y)                                \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }

#endif // AUTODIFF_SRC_EIGEN_CWISE_FACTORIES_HPP
