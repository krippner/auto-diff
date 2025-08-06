// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file concepts.hpp
 * @brief Defines concepts related to Eigen types.
 */

#ifndef AUTODIFF_SRC_EIGEN_CONCEPTS_HPP
#define AUTODIFF_SRC_EIGEN_CONCEPTS_HPP

#include "../Core/Expression.hpp" // ValueType

#include <concepts>    // integral floating_point
#include <type_traits> // is_base_of

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

template <typename T>
concept Scalar = std::integral<T> || std::floating_point<T>;

template <typename T>
concept MatrixBase = std::is_base_of_v<Eigen::MatrixBase<T>, T>;

template <typename T>
concept Dense = std::is_base_of_v<Eigen::DenseBase<T>, T>;

template <typename T>
concept Array = std::is_base_of_v<Eigen::ArrayBase<T>, T>;

template <typename T>
concept RowVector = MatrixBase<T> && T::RowsAtCompileTime == 1;

template <typename T>
concept ColVector = MatrixBase<T> && T::ColsAtCompileTime == 1;

template <typename T>
concept Matrix = MatrixBase<T> && !T::IsVectorAtCompileTime;

template <typename Expr>
concept ScalarExpression = Scalar<ValueType_t<Expr>>;

template <typename Expr>
concept ArrayExpression = Array<ValueType_t<Expr>>;

template <typename Expr>
concept MatrixExpression = Matrix<ValueType_t<Expr>>;

template <typename Expr>
concept RowVectorExpression = RowVector<ValueType_t<Expr>>;

template <typename Expr>
concept ColVectorExpression = ColVector<ValueType_t<Expr>>;

template <typename Expr>
concept MatrixBaseExpression = MatrixBase<ValueType_t<Expr>>;

} // namespace AutoDiff::EigenAD

#endif // AUTODIFF_SRC_EIGEN_CONCEPTS_HPP
