// Copyright (c) 2024 Matthias Krippner
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

#include "../traits.hpp"

#define AUTODIFF_MAKE_ARRAY_UNARY_OP(operation, Type)                          \
    template <typename X>                                                      \
    auto operation(Expression<X> const& x)                                     \
        -> std::enable_if_t<EigenAD::hasArrayValue_v<X>, Type<X>>              \
    {                                                                          \
        return Type<X>(x);                                                     \
    }

#define AUTODIFF_MAKE_ARRAY_ARRAY_OP(operation, Type)                          \
    template <typename X, typename Y>                                          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
        -> std::enable_if_t<EigenAD::hasArrayValue_v<X>                        \
                                && EigenAD::hasArrayValue_v<Y>,                \
            Type<X, Y>>                                                        \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }                                                                          \
                                                                               \
    template <typename Derived, typename Y>                                    \
    auto operation(Eigen::ArrayBase<Derived> const& x, Expression<Y> const& y) \
        -> std::enable_if_t<EigenAD::hasArrayValue_v<Y>, Type<Derived, Y>>     \
    {                                                                          \
        return Type<Derived, Y>(x.derived(), y);                               \
    }                                                                          \
                                                                               \
    template <typename X, typename Derived>                                    \
    auto operation(Expression<X> const& x, Eigen::ArrayBase<Derived> const& y) \
        -> std::enable_if_t<EigenAD::hasArrayValue_v<X>, Type<X, Derived>>     \
    {                                                                          \
        return Type<X, Derived>(x, y.derived());                               \
    }

#define AUTODIFF_MAKE_ARRAY_SCALAR_OP(operation, Type)                         \
    template <typename X, typename Scalar>                                     \
    auto operation(Expression<X> const& x,                                     \
        Scalar y) -> std::enable_if_t<EigenAD::hasArrayValue_v<X>              \
                                          && EigenAD::isScalar_v<Scalar>,      \
                      Type<X, Scalar>>                                         \
    {                                                                          \
        return Type<X, Scalar>(x, y);                                          \
    }                                                                          \
                                                                               \
    template <typename X, typename Y>                                          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
        -> std::enable_if_t<EigenAD::hasArrayValue_v<X>                        \
                                && EigenAD::hasScalarValue_v<Y>,               \
            Type<X, Y>>                                                        \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }

#define AUTODIFF_MAKE_SCALAR_ARRAY_OP(operation, Type)                         \
    template <typename Scalar, typename Y>                                     \
    auto operation(Scalar x, Expression<Y> const& y)                           \
        -> std::enable_if_t<EigenAD::isScalar_v<Scalar>                        \
                                && EigenAD::hasArrayValue_v<Y>,                \
            Type<Scalar, Y>>                                                   \
    {                                                                          \
        return Type<Scalar, Y>(x, y);                                          \
    }                                                                          \
                                                                               \
    template <typename X, typename Y>                                          \
    auto operation(Expression<X> const& x, Expression<Y> const& y)             \
        -> std::enable_if_t<EigenAD::hasScalarValue_v<X>                       \
                                && EigenAD::hasArrayValue_v<Y>,                \
            Type<X, Y>>                                                        \
    {                                                                          \
        return Type<X, Y>(x, y);                                               \
    }

#define AUTODIFF_MAKE_ARRAY_BINARY_OP(operation, Type)                         \
    AUTODIFF_MAKE_ARRAY_ARRAY_OP(operation, Type)                              \
    AUTODIFF_MAKE_ARRAY_SCALAR_OP(operation, Type)                             \
    AUTODIFF_MAKE_SCALAR_ARRAY_OP(operation, Type)

#endif // AUTODIFF_SRC_EIGEN_ARRAY_FACTORIES_HPP
