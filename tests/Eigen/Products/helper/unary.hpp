#ifndef TESTS_MODULES_EIGEN_PRODUCTS_HELPER_UNARY_HPP
#define TESTS_MODULES_EIGEN_PRODUCTS_HELPER_UNARY_HPP

#include "../../../helper/MockOperation.hpp"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <type_traits>

namespace detail {

template <typename T>
using Unqualified_t = std::remove_cv_t<std::remove_reference_t<T>>;

/**
 * @brief point = MatrixBase, value = MatrixBase
 */
template <typename Value, typename Derivative, typename Derived, typename P,
    typename V, typename D>
void checkUnaryOp(test::MockOperation<Value, Derivative>& operand,
    AutoDiff::Expression<Derived>& expression,
    Eigen::MatrixBase<P> const& point, Eigen::MatrixBase<V> const& targetValue,
    Eigen::MatrixBase<D> const& targetDeriv, double prec)
{
    operand.value() = point;
    auto const exprValue{
        expression._value()}; // evaluates expression, needed for diff
    THEN("operation yields correct value")
    {
        CAPTURE(exprValue, targetValue);
        CHECK(exprValue.isApprox(targetValue, prec));
    }
    auto const sizeP = point.size();
    auto const sizeV = targetValue.size();
    WHEN("pushing forward tangent")
    {
        operand.derivative() = Eigen::MatrixXd::Identity(sizeP, sizeP);
        auto const opDerivative{expression._pushForward()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(opDerivative, targetDeriv);
            CHECK(opDerivative.isApprox(targetDeriv, prec));
        }
    }
    WHEN("pulling back gradient")
    {
        operand.derivative() = Eigen::MatrixXd::Zero(sizeP, sizeP);
        expression._pullBack(Eigen::MatrixXd::Identity(sizeV, sizeV));
        auto const opDerivative{operand.derivative()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(opDerivative, targetDeriv);
            CHECK(opDerivative.isApprox(targetDeriv, prec));
        }
    }
    THEN("operation has correct derivative type")
    {
        static_assert(std::is_same_v<typename Derived::Derivative, Derivative>);
    }
}

/**
 * @brief point = MatrixBase, value = double
 */
template <typename Value, typename Derivative, typename Derived, typename P,
    typename D>
void checkUnaryOp(test::MockOperation<Value, Derivative>& operand,
    AutoDiff::Expression<Derived>& expression,
    Eigen::MatrixBase<P> const& point, double targetValue,
    Eigen::MatrixBase<D> const& targetDeriv, double prec)
{
    using Catch::Matchers::WithinAbsMatcher;
    operand.value() = point;
    auto const exprValue{
        expression._value()}; // evaluates expression, needed for diff
    THEN("operation yields correct value")
    {
        CHECK_THAT(exprValue, WithinAbsMatcher(targetValue, prec));
    }
    auto const size = point.size();
    WHEN("pushing forward tangent")
    {
        operand.derivative() = Eigen::MatrixXd::Identity(size, size);
        auto const opDerivative{expression._pushForward()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(opDerivative, targetDeriv);
            CHECK(opDerivative.isApprox(targetDeriv, prec));
        }
    }
    WHEN("pulling back gradient")
    {
        operand.derivative() = Eigen::RowVectorXd::Zero(size);
        expression._pullBack(Eigen::MatrixXd::Identity(1, 1));
        auto const opDerivative{operand.derivative()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(opDerivative, targetDeriv);
            CHECK(opDerivative.isApprox(targetDeriv, prec));
        }
    }
}

} // namespace detail

/**
 * @brief Checks whether, given point p, the unitary operation yields
 * the specified value and derivative within a margin.
 */
#define CHECK_UNARY_OP(operation, p, v, d, prec)                               \
    WHEN("evaluating")                                                         \
    {                                                                          \
        using Point     = detail::Unqualified_t<decltype(p)>;                  \
        auto operand    = test::MockOperation<Point, Eigen::MatrixXd>();       \
        auto expression = operation(operand);                                  \
        detail::checkUnaryOp(operand, expression, p, v, d, prec);              \
    }

#endif // TESTS_MODULES_EIGEN_PRODUCTS_HELPER_UNARY_HPP
