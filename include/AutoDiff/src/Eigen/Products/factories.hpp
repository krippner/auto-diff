// Copyright (c) 2024 Matthias Krippner
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

#include "../traits.hpp"

#define AUTODIFF_MAKE_MATRIXBASE_BINARY_OP(operation, Type)                    \
    template <typename X, typename Y>                                          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
        -> std::enable_if_t<EigenAD::hasMatrixBaseValue_v<X>                   \
                                && EigenAD::hasMatrixBaseValue_v<Y>,           \
            Type<X, Y>>                                                        \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Derived, typename Y>                                    \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
        -> std::enable_if_t<EigenAD::hasMatrixBaseValue_v<Y>,                  \
            Type<Derived, Y>>                                                  \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <typename X, typename Derived>                                    \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
        -> std::enable_if_t<EigenAD::hasMatrixBaseValue_v<X>,                  \
            Type<X, Derived>>                                                  \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#define AUTODIFF_MAKE_COLVECTOR_BINARY_OP(operation, Type)                     \
    template <typename X, typename Y>                                          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
        -> std::enable_if_t<EigenAD::hasColVectorValue_v<X>                    \
                                && EigenAD::hasColVectorValue_v<Y>,            \
            Type<X, Y>>                                                        \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Derived, typename Y>                                    \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
        -> std::enable_if_t<EigenAD::isColVector_v<Derived>                    \
                                && EigenAD::hasColVectorValue_v<Y>,            \
            Type<Derived, Y>>                                                  \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <typename X, typename Derived>                                    \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
        -> std::enable_if_t<EigenAD::hasColVectorValue_v<X>                    \
                                && EigenAD::isColVector_v<Derived>,            \
            Type<X, Derived>>                                                  \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#define AUTODIFF_MAKE_MATRIX_BINARY_OP(operation, Type)                        \
    template <typename X, typename Y>                                          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
        -> std::enable_if_t<EigenAD::hasMatrixValue_v<X>                       \
                                && EigenAD::hasMatrixValue_v<Y>,               \
            Type<X, Y>>                                                        \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Derived, typename Y>                                    \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
        -> std::enable_if_t<EigenAD::isMatrix_v<Derived>                       \
                                && EigenAD::hasMatrixValue_v<Y>,               \
            Type<Derived, Y>>                                                  \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <typename X, typename Derived>                                    \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
        -> std::enable_if_t<EigenAD::hasMatrixValue_v<X>                       \
                                && EigenAD::isMatrix_v<Derived>,               \
            Type<X, Derived>>                                                  \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#define AUTODIFF_MAKE_MATRIX_COLVECTOR_OP(operation, Type)                     \
    template <typename X, typename Y>                                          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
        -> std::enable_if_t<EigenAD::hasMatrixValue_v<X>                       \
                                && EigenAD::hasColVectorValue_v<Y>,            \
            Type<X, Y>>                                                        \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Derived, typename Y>                                    \
    auto operation(                                                            \
        Eigen::MatrixBase<Derived> const& x, Expression<Y> const& y)           \
        -> std::enable_if_t<EigenAD::isMatrix_v<Derived>                       \
                                && EigenAD::hasColVectorValue_v<Y>,            \
            Type<Derived, Y>>                                                  \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <typename X, typename Derived>                                    \
    auto operation(                                                            \
        Expression<X> const& x, Eigen::MatrixBase<Derived> const& y)           \
        -> std::enable_if_t<EigenAD::hasMatrixBaseValue_v<X>                   \
                                && EigenAD::isColVector_v<Derived>,            \
            Type<X, Derived>>                                                  \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#endif // AUTODIFF_SRC_EIGEN_PRODUCTS_FACTORIES_HPP
