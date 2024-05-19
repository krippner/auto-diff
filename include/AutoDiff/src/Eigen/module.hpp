// Copyright (c) 2024 Matthias Krippner
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
#include "traits.hpp"             // isScalar, isDense, isMatrixBase, isArray

#include <type_traits>
#include <utility> // declval

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

namespace AutoDiff::detail {

template <typename T>
using Unqualified_t = std::remove_cv_t<std::remove_reference_t<T>>;

} // namespace AutoDiff::detail

namespace AutoDiff::internal {

// Scalar types are already equal to their evaluated types.
template <typename Scalar>
struct Evaluated<Scalar, std::enable_if_t<EigenAD::isScalar_v<Scalar>>> {
    using type = Scalar;
};

// Let Eigen decide the evaluated type of dense Eigen types.
template <typename Dense>
struct Evaluated<Dense, std::enable_if_t<EigenAD::isDense_v<Dense>>> {
    using type = detail::Unqualified_t<decltype(std::declval<Dense>().eval())>;
};

// For arrays, values and derivatives must have the same type
// (or at least same dimensions at runtime).
template <typename Array>
struct DefaultDerivative<Array, std::enable_if_t<EigenAD::isArray_v<Array>>> {
    using type = Array;
};

// By default, floats are paired with float derivatives...

template <>
struct DefaultDerivative<float> {
    using type = Eigen::MatrixXf;
};

template <typename Matrix>
struct DefaultDerivative<Matrix,
    std::enable_if_t<EigenAD::isMatrixBase_v<Matrix>
                     && std::is_same_v<typename Matrix::Scalar, float>>> {
    using type = Eigen::MatrixXf;
};

// ...otherwise, use double derivatives.

template <typename T>
struct DefaultDerivative<T,
    std::enable_if_t<EigenAD::isScalar_v<T> && !std::is_same_v<T, float>>> {
    using type = Eigen::MatrixXd;
};

template <typename T>
struct DefaultDerivative<T,
    std::enable_if_t<EigenAD::isMatrixBase_v<T>
                     && !std::is_same_v<typename T::Scalar, float>>> {
    using type = Eigen::MatrixXd;
};

} // namespace AutoDiff::internal

// implementation of type-specific operations

namespace AutoDiff::internal {

template <typename Scalar>
struct TypeImpl<Scalar, std::enable_if_t<EigenAD::isScalar_v<Scalar>>> {
    static auto getShape(Scalar const& /*scalar*/) -> Shape { return {1}; }
    static void assign(Scalar& value, Scalar const& other) { value = other; }
};

template <typename MatrixBase>
struct TypeImpl<MatrixBase,
    std::enable_if_t<EigenAD::isMatrixBase_v<MatrixBase>>> {
    static auto getShape(MatrixBase const& matrix) -> Shape
    {
        return {static_cast<std::size_t>(matrix.size())};
    }

    static auto codomainShape(MatrixBase const& matrix) -> Shape
    {
        return {static_cast<std::size_t>(matrix.rows())};
    }

    static void generate(MatrixBase& matrix, MapDescription const& descr)
    {
        if (descr.state == MapDescription::zero) {
            matrix.setZero(descr.codomainShape[0], descr.domainShape[0]);
        } else if (descr.state == MapDescription::identity) {
            matrix.setIdentity(descr.codomainShape[0], descr.domainShape[0]);
        }
    }

    template <typename Other>
    static void assign(MatrixBase& matrix, Other const& other)
    {
        matrix.noalias() = other;
    }

    template <typename Other>
    static void addTo(MatrixBase& matrix, Other const& other)
    {
        matrix.noalias() += other;
    }
};

template <typename Array>
struct TypeImpl<Array, std::enable_if_t<EigenAD::isArray_v<Array>>> {
    static auto getShape(Array const& array) -> Shape
    {
        return {static_cast<std::size_t>(array.rows()),
            static_cast<std::size_t>(array.cols())};
    }

    static auto codomainShape(Array const& array) -> Shape
    {
        return {static_cast<std::size_t>(array.rows()),
            static_cast<std::size_t>(array.cols())};
    }

    static void generate(Array& array, MapDescription const& descr)
    {
        if (descr.state == MapDescription::zero) {
            array.setZero(descr.domainShape[0], descr.domainShape[1]);
        } else if (descr.state == MapDescription::identity) {
            array.setOnes(descr.domainShape[0], descr.domainShape[1]);
        }
    }

    template <typename Other>
    static void assign(Array& array, Other const& other)
    {
        array = other;
    }

    template <typename Other>
    static void addTo(Array& array, Other const& other)
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
using Integer = Variable<int, Eigen::MatrixXd>;
using Boolean = Variable<bool, Eigen::MatrixXd>;

using RealF    = Variable<float, Eigen::MatrixXf>;
using IntegerF = Variable<int, Eigen::MatrixXf>;
using BooleanF = Variable<bool, Eigen::MatrixXf>;

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
