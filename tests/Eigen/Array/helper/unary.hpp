#ifndef TESTS_MODULES_EIGEN_ARRAY_HELPER_UNARY_HPP
#define TESTS_MODULES_EIGEN_ARRAY_HELPER_UNARY_HPP

#include "../../../helper/MockOperation.hpp"

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>

#include <type_traits>

namespace detail {

template <typename T>
using Unqualified_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename Value, typename Derivative, typename Derived, typename P,
    typename V, typename D>
void checkUnaryOp(test::MockOperation<Value, Derivative>& operand,
    AutoDiff::Expression<Derived>& expression, Eigen::ArrayBase<P> const& point,
    Eigen::ArrayBase<V> const& targetValue,
    Eigen::ArrayBase<D> const& targetDeriv, double prec)
{
    operand.value() = point;
    auto const exprValue{
        expression._value()}; // evaluates expression, needed for diff
    THEN("operation yields correct value")
    {
        CAPTURE(exprValue, targetValue);
        CHECK(exprValue.isApprox(targetValue, prec));
    }
    auto const rows = targetValue.rows();
    auto const cols = targetValue.cols();
    WHEN("pushing forward tangent")
    {
        operand.derivative() = Eigen::ArrayXXd::Ones(rows, cols);
        auto const opDerivative{expression._pushForward()};
        THEN("operation yields correct derivative")
        {
            CAPTURE(opDerivative, targetDeriv);
            CHECK(opDerivative.isApprox(targetDeriv, prec));
        }
    }
    WHEN("pulling back gradient")
    {
        operand.derivative() = Eigen::ArrayXXd::Zero(rows, cols);
        expression._pullBack(Eigen::ArrayXXd::Ones(rows, cols));
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

} // namespace detail

/**
 * @brief Checks whether, given point p, the unitary operation yields
 * the specified value and derivative within a margin.
 */
#define CHECK_UNARY_OP(operation, p, v, d, prec)                               \
    WHEN("evaluating")                                                         \
    {                                                                          \
        using Point     = detail::Unqualified_t<decltype(p)>;                  \
        auto operand    = test::MockOperation<Point, Eigen::ArrayXXd>();       \
        auto expression = operation(operand);                                  \
        detail::checkUnaryOp(operand, expression, p, v, d, prec);              \
    }

#endif // TESTS_MODULES_EIGEN_ARRAY_HELPER_UNARY_HPP
