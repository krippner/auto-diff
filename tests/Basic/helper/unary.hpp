#ifndef TESTS_BASIC_HELPER_UNARY_HPP
#define TESTS_BASIC_HELPER_UNARY_HPP

#include "../../helper/MockOperation.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <type_traits>

namespace detail {

template <typename T>
using Unqualified_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename Value, typename Derivative, typename Derived>
void checkUnaryOp(test::MockOperation<Value, Derivative>& operand,
    AutoDiff::Expression<Derived>& expression, double point, double targetValue,
    double targetDeriv, double prec)
{
    using Catch::Matchers::WithinAbsMatcher;
    operand.value() = point;
    auto const exprValue{expression._value()};
    THEN("operation yields correct value")
    {
        CHECK_THAT(exprValue, WithinAbsMatcher(targetValue, prec));
    }
    WHEN("pushing forward tangent")
    {
        operand.derivative() = 1.0;
        auto const opDerivative{expression._pushForward()};
        THEN("operation yields correct derivative")
        {
            CHECK_THAT(opDerivative, WithinAbsMatcher(targetDeriv, prec));
        }
    }
    WHEN("pulling back gradient")
    {
        expression._pullBack(1.0);
        auto const opDerivative{operand.derivative()};
        THEN("operation yields correct derivative")
        {
            CHECK_THAT(opDerivative, WithinAbsMatcher(targetDeriv, prec));
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
        auto operand    = test::MockOperation<Point, double>();                \
        auto expression = operation(operand);                                  \
        detail::checkUnaryOp(operand, expression, p, v, d, prec);              \
    }

#endif // TESTS_BASIC_HELPER_UNARY_HPP
