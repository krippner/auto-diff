// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file module.hpp
 * @brief Defines necessary type traits to enable AutoDiff for Eigen types.
 *
 * Additionally, this file provides aliases for common @c Variable types.
 */

#ifndef AUTODIFF_SRC_EIGEN_MODULE_HPP
#define AUTODIFF_SRC_EIGEN_MODULE_HPP

#define AUTODIFF_MODULE // only one module per translation unit

#include "../internal/TypeImpl.hpp"
#include "../internal/traits.hpp" // traits to be specialized
#include "concepts.hpp"           // Scalar, Dense, MatrixBase, Array

#include <type_traits> // remove_cvref is_same conditional
#include <utility>     // declval

// forward-declare Eigen types

namespace Eigen {

template <typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_,
    int MaxCols_>
class Array;

using ArrayXXd = Array<double, -1, -1, 0, -1, -1>;
using ArrayXd  = Array<double, -1, 1, 0, -1, 1>;
using ArrayXXf = Array<float, -1, -1, 0, -1, -1>;
using ArrayXf  = Array<float, -1, 1, 0, -1, 1>;

template <typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_,
    int MaxCols_>
class Matrix;

using MatrixXd = Matrix<double, -1, -1, 0, -1, -1>;
using Matrix2d = Matrix<double, 2, 2, 0, 2, 2>;
using Matrix3d = Matrix<double, 3, 3, 0, 3, 3>;
using Matrix4d = Matrix<double, 4, 4, 0, 4, 4>;
using MatrixXf = Matrix<float, -1, -1, 0, -1, -1>;
using Matrix2f = Matrix<float, 2, 2, 0, 2, 2>;
using Matrix3f = Matrix<float, 3, 3, 0, 3, 3>;
using Matrix4f = Matrix<float, 4, 4, 0, 4, 4>;

using VectorXd = Matrix<double, -1, 1, 0, -1, 1>;
using Vector2d = Matrix<double, 2, 1, 0, 2, 1>;
using Vector3d = Matrix<double, 3, 1, 0, 3, 1>;
using Vector4d = Matrix<double, 4, 1, 0, 4, 1>;
using VectorXf = Matrix<float, -1, 1, 0, -1, 1>;
using Vector2f = Matrix<float, 2, 1, 0, 2, 1>;
using Vector3f = Matrix<float, 3, 1, 0, 3, 1>;
using Vector4f = Matrix<float, 4, 1, 0, 4, 1>;

} // namespace Eigen

// mandatory specializations of type traits for Eigen types

namespace AutoDiff::internal {

// Scalar types are already equal to their evaluated types.
template <EigenAD::Scalar T>
struct Evaluated<T> {
    using type = T;
};

// Let Eigen decide the evaluated type of dense Eigen types.
template <EigenAD::Dense T>
struct Evaluated<T> {
    using type = std::remove_cvref_t<decltype(std::declval<T>().eval())>;
};

template <EigenAD::Array T>
struct DefaultDerivative<T> {
    using Scalar = std::conditional_t<std::is_same_v<typename T::Scalar, float>,
        float, // float -> float derivative
        double // otherwise
        >;
    // Arrays are paired with derivatives of same shape
    using type = Eigen::Array<Scalar, //
        T::RowsAtCompileTime,         //
        T::ColsAtCompileTime,         //
        0,                            //
        T::MaxRowsAtCompileTime,      //
        T::MaxColsAtCompileTime>;
};

template <EigenAD::Scalar T>
struct DefaultDerivative<T> {
    using type = std::conditional_t<std::is_same_v<T, float>,
        Eigen::MatrixXf, // float -> float derivative
        Eigen::MatrixXd  // otherwise
        >;
};

template <EigenAD::MatrixBase T>
struct DefaultDerivative<T> {
    using type = std::conditional_t<std::is_same_v<typename T::Scalar, float>,
        Eigen::MatrixXf, // float -> float derivative
        Eigen::MatrixXd  // otherwise
        >;
};

} // namespace AutoDiff::internal

// implementation of type-specific operations

namespace AutoDiff::internal {

template <EigenAD::Scalar T>
struct TypeImpl<T> {
    static auto getShape(T const& /*scalar*/) -> Shape { return {1}; }
    static void assign(T& value, T const& other) { value = other; }
};

template <EigenAD::MatrixBase T>
struct TypeImpl<T> {
    static auto getShape(T const& matrix) -> Shape
    {
        return {static_cast<std::size_t>(matrix.size())};
    }

    static auto codomainShape(T const& matrix) -> Shape
    {
        return {static_cast<std::size_t>(matrix.rows())};
    }

    static void generate(T& matrix, MapDescription const& descr)
    {
        if (descr.state == MapDescription::zero) {
            matrix.setZero(descr.codomainShape[0], descr.domainShape[0]);
        } else if (descr.state == MapDescription::identity) {
            matrix.setIdentity(descr.codomainShape[0], descr.domainShape[0]);
        }
    }

    template <typename Other>
    static void assign(T& matrix, Other const& other)
    {
        matrix.noalias() = other;
    }

    template <typename Other>
    static void addTo(T& matrix, Other const& other)
    {
        matrix.noalias() += other;
    }
};

template <EigenAD::Array T>
struct TypeImpl<T> {
    static auto getShape(T const& array) -> Shape
    {
        return {static_cast<std::size_t>(array.rows()),
            static_cast<std::size_t>(array.cols())};
    }

    static auto codomainShape(T const& array) -> Shape
    {
        return {static_cast<std::size_t>(array.rows()),
            static_cast<std::size_t>(array.cols())};
    }

    static void generate(T& array, MapDescription const& descr)
    {
        if (descr.state == MapDescription::zero) {
            array.setZero(descr.domainShape[0], descr.domainShape[1]);
        } else if (descr.state == MapDescription::identity) {
            array.setOnes(descr.domainShape[0], descr.domainShape[1]);
        }
    }

    template <typename Other>
    static void assign(T& array, Other const& other)
    {
        array = other;
    }

    template <typename Other>
    static void addTo(T& array, Other const& other)
    {
        array += other;
    }
};

} // namespace AutoDiff::internal

// aliases

namespace AutoDiff {

template <typename Value, typename Derivative>
class Variable;

using Real    = Variable<double, Eigen::MatrixXd>;
using RealF   = Variable<float, Eigen::MatrixXf>;
using Integer = Variable<int, Eigen::MatrixXd>;
using Boolean = Variable<bool, Eigen::MatrixXd>;

using Vector   = Variable<Eigen::VectorXd, Eigen::MatrixXd>;
using Vector2d = Variable<Eigen::Vector2d, Eigen::MatrixXd>;
using Vector3d = Variable<Eigen::Vector3d, Eigen::MatrixXd>;
using Vector4d = Variable<Eigen::Vector4d, Eigen::MatrixXd>;
using Matrix   = Variable<Eigen::MatrixXd, Eigen::MatrixXd>;
using Matrix2d = Variable<Eigen::Matrix2d, Eigen::MatrixXd>;
using Matrix3d = Variable<Eigen::Matrix3d, Eigen::MatrixXd>;
using Matrix4d = Variable<Eigen::Matrix4d, Eigen::MatrixXd>;

using VectorXf = Variable<Eigen::VectorXf, Eigen::MatrixXf>;
using Vector2f = Variable<Eigen::Vector2f, Eigen::MatrixXf>;
using Vector3f = Variable<Eigen::Vector3f, Eigen::MatrixXf>;
using Vector4f = Variable<Eigen::Vector4f, Eigen::MatrixXf>;
using MatrixXf = Variable<Eigen::MatrixXf, Eigen::MatrixXf>;
using Matrix2f = Variable<Eigen::Matrix2f, Eigen::MatrixXf>;
using Matrix3f = Variable<Eigen::Matrix3f, Eigen::MatrixXf>;
using Matrix4f = Variable<Eigen::Matrix4f, Eigen::MatrixXf>;

using Array   = Variable<Eigen::ArrayXd, Eigen::ArrayXd>;
using ArrayXX = Variable<Eigen::ArrayXXd, Eigen::ArrayXXd>;

using ArrayXf  = Variable<Eigen::ArrayXf, Eigen::ArrayXf>;
using ArrayXXf = Variable<Eigen::ArrayXXf, Eigen::ArrayXXf>;

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_MODULE_HPP
