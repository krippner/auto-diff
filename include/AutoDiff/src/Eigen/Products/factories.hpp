// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file factories.hpp
 * @brief Macros defining factory functions for products between Eigen matrices.
 *
 * Use these macros inside the @c AutoDiff namespace. They can be used without
 * namespace qualification in user code because of argument-dependent lookup.
 */

#ifndef AUTODIFF_SRC_EIGEN_PRODUCTS_FACTORIES_HPP
#define AUTODIFF_SRC_EIGEN_PRODUCTS_FACTORIES_HPP

#include "../concepts.hpp"

#define AUTODIFF_MAKE_MATRIXBASE_BINARY_OP(operation, Type)                    \
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

#define AUTODIFF_MAKE_COLVECTOR_BINARY_OP(operation, Type)                     \
    template <EigenAD::ColVectorExpression X, EigenAD::ColVectorExpression Y>  \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <EigenAD::ColVector Derived, EigenAD::ColVectorExpression Y>      \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <EigenAD::ColVectorExpression X, EigenAD::ColVector Derived>      \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#define AUTODIFF_MAKE_MATRIX_BINARY_OP(operation, Type)                        \
    template <EigenAD::MatrixExpression X, EigenAD::MatrixExpression Y>        \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <EigenAD::Matrix Derived, EigenAD::MatrixExpression Y>            \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <EigenAD::MatrixExpression X, EigenAD::Matrix Derived>            \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#define AUTODIFF_MAKE_MATRIX_COLVECTOR_OP(operation, Type)                     \
    template <EigenAD::MatrixExpression X, EigenAD::ColVectorExpression Y>     \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <EigenAD::Matrix Derived, EigenAD::ColVectorExpression Y>         \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <EigenAD::MatrixExpression X, EigenAD::ColVector Derived>         \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#endif // AUTODIFF_SRC_EIGEN_PRODUCTS_FACTORIES_HPP
