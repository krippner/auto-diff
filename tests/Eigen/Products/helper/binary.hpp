#ifndef TESTS_MODULES_EIGEN_PRODUCTS_HELPER_BINARY_HPP
#define TESTS_MODULES_EIGEN_PRODUCTS_HELPER_BINARY_HPP

#include "../../../helper/MockOperation.hpp"
#include "unary.hpp"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <type_traits>

namespace detail {

template <typename T>
using Unqualified_t = std::remove_cv_t<std::remove_reference_t<T>>;

/**
 * @brief pointX = MatrixBase, pointY = MatrixBase, value = MatrixBase
 */
template <typename ValueX, typename ValueY, typename Derivative,
    typename Derived, typename X, typename Y, typename V, typename DX,
    typename DY>
void checkBinaryOp(test::MockOperation<ValueX, Derivative>& operandX,
    test::MockOperation<ValueY, Derivative>& operandY,
    AutoDiff::Expression<Derived>& expression,
    Eigen::MatrixBase<X> const& pointX, Eigen::MatrixBase<Y> const& pointY,
    Eigen::MatrixBase<V> const& targetValue,
    Eigen::MatrixBase<DX> const& targetDerivX,
    Eigen::MatrixBase<DY> const& targetDerivY, double prec)
{
    operandX.value() = pointX;
    operandY.value() = pointY;
    auto const exprValue{
        expression._value()}; // evaluates expression, needed for diff
    THEN("operation yields correct value")
    {
        CAPTURE(exprValue, targetValue);
        CHECK(exprValue.isApprox(targetValue, prec));
    }
    auto const sizeX = pointX.size();
    auto const sizeY = pointY.size();
    auto const sizeV = targetValue.size();
    WHEN("pushing forward tangent from x")
    {
        operandX.derivative() = Eigen::MatrixXd::Identity(sizeX, sizeX);
        operandY.derivative() = Eigen::MatrixXd::Zero(sizeY, sizeX);
        auto const derivativeX{expression._pushForward()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(derivativeX, targetDerivX);
            CHECK(derivativeX.isApprox(targetDerivX, prec));
        }
    }
    WHEN("pushing forward tangent from y")
    {
        operandX.derivative() = Eigen::MatrixXd::Zero(sizeX, sizeY);
        operandY.derivative() = Eigen::MatrixXd::Identity(sizeY, sizeY);
        auto const derivativeY{expression._pushForward()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(derivativeY, targetDerivY);
            CHECK(derivativeY.isApprox(targetDerivY, prec));
        }
    }
    WHEN("pulling back gradient")
    {
        expression._pullBack(Eigen::MatrixXd::Identity(sizeV, sizeV));
        auto const derivativeX{operandX.derivative()};
        auto const derivativeY{operandY.derivative()};
        THEN("operation yields correct derivatives")
        {
            CAPTURE(derivativeX, targetDerivX);
            CHECK(derivativeX.isApprox(targetDerivX, prec));
            CAPTURE(derivativeY, targetDerivY);
            CHECK(derivativeY.isApprox(targetDerivY, prec));
        }
    }
    THEN("operation has correct derivative type")
    {
        static_assert(std::is_same_v<typename Derived::Derivative, Derivative>);
    }
}

/**
 * @brief pointX = MatrixBase, pointY = MatrixBase, value = double
 */
template <typename ValueX, typename ValueY, typename Derivative,
    typename Derived, typename X, typename Y, typename DX, typename DY>
void checkBinaryOp(test::MockOperation<ValueX, Derivative>& operandX,
    test::MockOperation<ValueY, Derivative>& operandY,
    AutoDiff::Expression<Derived>& expression,
    Eigen::MatrixBase<X> const& pointX, Eigen::MatrixBase<Y> const& pointY,
    double targetValue, Eigen::MatrixBase<DX> const& targetDerivX,
    Eigen::MatrixBase<DY> const& targetDerivY, double prec)
{
    using Catch::Matchers::WithinAbsMatcher;
    operandX.value() = pointX;
    operandY.value() = pointY;
    auto const exprValue{
        expression._value()}; // evaluates expression, needed for diff
    THEN("operation yields correct value")
    {
        CHECK_THAT(exprValue, WithinAbsMatcher(targetValue, prec));
    }
    auto const sizeX = pointX.size();
    auto const sizeY = pointY.size();
    WHEN("pushing forward tangent from x")
    {
        operandX.derivative() = Eigen::MatrixXd::Identity(sizeX, sizeX);
        operandY.derivative() = Eigen::MatrixXd::Zero(sizeY, sizeY);
        auto const derivativeX{expression._pushForward()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(derivativeX, targetDerivX);
            CHECK(derivativeX.isApprox(targetDerivX, prec));
        }
    }
    WHEN("pushing forward tangent from y")
    {
        operandX.derivative() = Eigen::MatrixXd::Zero(sizeX, sizeX);
        operandY.derivative() = Eigen::MatrixXd::Identity(sizeY, sizeY);
        auto const derivativeY{expression._pushForward()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(derivativeY, targetDerivY);
            CHECK(derivativeY.isApprox(targetDerivY, prec));
        }
    }
    WHEN("pulling back gradient")
    {
        expression._pullBack(Eigen::MatrixXd::Identity(1, 1));
        auto const derivativeX{operandX.derivative()};
        auto const derivativeY{operandY.derivative()};
        THEN("operation yields correct derivatives")
        {
            CAPTURE(derivativeX, targetDerivX);
            CHECK(derivativeX.isApprox(targetDerivX, prec));
            CAPTURE(derivativeY, targetDerivY);
            CHECK(derivativeY.isApprox(targetDerivY, prec));
        }
    }
}

} // namespace detail

/**
 * @brief Checks whether, given point (pX, pY), the binary operation yields
 * the specified value and derivatives within a margin.
 */
#define CHECK_BINARY_OP(operation, pX, pY, v, dX, dY, prec)                    \
    WHEN("evaluating")                                                         \
    {                                                                          \
        using PointX    = detail::Unqualified_t<decltype(pX)>;                 \
        using PointY    = detail::Unqualified_t<decltype(pY)>;                 \
        auto operandX   = test::MockOperation<PointX, Eigen::MatrixXd>();      \
        auto operandY   = test::MockOperation<PointY, Eigen::MatrixXd>();      \
        auto expression = operation(operandX, operandY);                       \
        detail::checkBinaryOp(                                                 \
            operandX, operandY, expression, pX, pY, v, dX, dY, prec);          \
    }                                                                          \
    WHEN("evaluating with left literal operand")                               \
    {                                                                          \
        using Point     = detail::Unqualified_t<decltype(pY)>;                 \
        auto operand    = test::MockOperation<Point, Eigen::MatrixXd>();       \
        auto expression = operation(pX, operand);                              \
        detail::checkUnaryOp(operand, expression, pY, v, dY, prec);            \
    }                                                                          \
    WHEN("evaluating with right literal operand")                              \
    {                                                                          \
        using Point     = detail::Unqualified_t<decltype(pX)>;                 \
        auto operand    = test::MockOperation<Point, Eigen::MatrixXd>();       \
        auto expression = operation(operand, pY);                              \
        detail::checkUnaryOp(operand, expression, pX, v, dX, prec);            \
    }

#endif // TESTS_MODULES_EIGEN_PRODUCTS_HELPER_BINARY_HPP
