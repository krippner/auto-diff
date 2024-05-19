// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file traits.hpp
 * @brief Defines traits that determine whether and which Eigen types are used.
 */

#ifndef AUTODIFF_SRC_EIGEN_TRAITS_HPP
#define AUTODIFF_SRC_EIGEN_TRAITS_HPP

#include "../Core/Expression.hpp" // ValueType

#include <type_traits> // is_arithmetic is_base_of true_type false_type

// forward-declare Eigen types

namespace Eigen {

template <typename Derived>
class DenseBase;

template <typename Derived>
class ArrayBase;

template <typename Derived>
class MatrixBase;

} // namespace Eigen

namespace AutoDiff::EigenAD {

namespace detail {

    template <typename Derived>
    auto testColVector(Eigen::MatrixBase<Derived> const* /*base*/)
    {
        if constexpr (Derived::ColsAtCompileTime == 1) {
            return std::true_type();
        } else {
            return std::false_type();
        }
    }
    auto testColVector(void const*) -> std::false_type;

    template <typename Derived>
    auto testRowVector(Eigen::MatrixBase<Derived> const* /*base*/)
    {
        if constexpr (Derived::RowsAtCompileTime == 1) {
            return std::true_type();
        } else {
            return std::false_type();
        }
    }
    auto testRowVector(void const*) -> std::false_type;

    template <typename Derived>
    auto testMatrix(Eigen::MatrixBase<Derived> const* /*base*/)
    {
        if constexpr (!Derived::IsVectorAtCompileTime) {
            return std::true_type();
        } else {
            return std::false_type();
        }
    }
    auto testMatrix(void const*) -> std::false_type;

} // namespace detail

template <typename T>
constexpr bool isScalar_v = std::is_arithmetic_v<T>;

template <typename T>
constexpr bool isDense_v = std::is_base_of_v<Eigen::DenseBase<T>, T>;

template <typename T>
constexpr bool isArray_v = std::is_base_of_v<Eigen::ArrayBase<T>, T>;

template <typename T>
constexpr bool isMatrixBase_v = std::is_base_of_v<Eigen::MatrixBase<T>, T>;

template <typename T>
constexpr bool isColVector_v
    = decltype(detail::testColVector(std::declval<T*>()))::value;

template <typename T>
constexpr bool isRowVector_v
    = decltype(detail::testRowVector(std::declval<T*>()))::value;

template <typename T>
constexpr bool isMatrix_v
    = decltype(detail::testMatrix(std::declval<T*>()))::value;

template <typename Expr>
constexpr bool hasScalarValue_v = isScalar_v<ValueType_t<Expr>>;

template <typename Expr>
constexpr bool hasArrayValue_v = isArray_v<ValueType_t<Expr>>;

template <typename Expr>
constexpr bool hasMatrixBaseValue_v = isMatrixBase_v<ValueType_t<Expr>>;

template <typename Expr>
constexpr bool hasColVectorValue_v = isColVector_v<ValueType_t<Expr>>;

template <typename Expr>
constexpr bool hasRowVectorValue_v = isRowVector_v<ValueType_t<Expr>>;

template <typename Expr>
constexpr bool hasMatrixValue_v = isMatrix_v<ValueType_t<Expr>>;

} // namespace AutoDiff::EigenAD

#endif // AUTODIFF_SRC_EIGEN_TRAITS_HPP
