#include "../helper/MockOperation.hpp"
#include "../helper/int.hpp"
#include "helper/MockJacobian.hpp"

#include <AutoDiff/src/internal/Computation.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp> // take
#include <catch2/generators/catch_generators_random.hpp>

using AutoDiff::internal::Computation;
using MockOperation = test::MockOperation<int, int>;

SCENARIO("Literal", "[Computation]")
{
    GIVEN("a default-constructed literal")
    {
        auto literal = Computation<int, int>();
        THEN("literal has default value and derivative")
        {
            CHECK(literal.value() == 0);
            CHECK(literal.derivative() == 0);
        }
        WHEN("setting the value")
        {
            auto const value = GENERATE(take(1, random(1, 10)));
            literal.setValue(value);
            THEN("value is cached") { CHECK(literal.value() == value); }
            // evaluation not allowed
        }
        WHEN("setting a custom derivative")
        {
            auto const derivative = GENERATE(take(1, random(1, 10)));
            literal.setDerivative(derivative);
            THEN("derivative is cached")
            {
                CHECK(literal.derivative() == derivative);
            }
        }
        WHEN("setting the derivative to identity")
        {
            literal.setDerivativeIdentity();
            THEN("identity derivative is generated")
            {
                CHECK(literal.derivative() == 1);
            }
        }
    }
}

SCENARIO("Expression evaluation", "[Computation]")
{
    GIVEN("an expression with some value and derivative")
    {
        auto const value        = GENERATE(take(1, random(1, 10)));
        auto const tangent      = GENERATE(take(1, random(1, 10)));
        auto expression         = MockOperation();
        expression.value()      = value;
        expression.derivative() = tangent;
        GIVEN("an evaluator of this expression")
        {
            auto evaluator = Computation<int, int>();
            evaluator.setExpression(expression);
            WHEN("evaluating")
            {
                evaluator.evaluate();
                CHECK(evaluator.value() == value);
            }
            WHEN("pushing forward the tangent")
            {
                evaluator.pushTangent();
                CHECK(evaluator.derivative() == tangent);
            }
            WHEN("pulling back a gradient")
            {
                auto const gradient = GENERATE(take(1, random(1, 10)));
                evaluator.setDerivative(gradient);
                evaluator.pullGradient();
                CHECK(expression.derivative() == gradient);
            }
        }
    }
}

// enable double type for use in Computation<double,..>

namespace AutoDiff::internal {

template <>
struct TypeImpl<double> {
    static auto getShape(double const& /*value*/) -> Shape { return {1}; }
};

} // namespace AutoDiff::internal

SCENARIO("Incompatible expression assignment", "[Computation]")
{
    auto expression = MockOperation();
    auto evaluator  = Computation<double, int>();
    // evaluator.setExpression(expression); // static_assert should fail
}

// test propagation of domain shapes ------------------------------------------

using AutoDiff::internal::AbstractComputation;
using AutoDiff::internal::Shape;

SCENARIO("Pushforward of tangent by f: A -> B", "[Computation]")
{
    /*
       Note:
       A tangent vector is represented by a column vector.
       More generally, we can assign a number of columns to the tangent,
       giving it the shape of a matrix.
       The pushforward always has the same number of columns.
     */

    // number of columns in tangent
    auto const cols = GENERATE(take(1, random<std::size_t>(1, 10)));

    // computation of a point b in B
    auto b = Computation<test::Point, test::Jacobian>();
    // some functions must be accessible through the base class
    auto& bBaseRef = dynamic_cast<AbstractComputation&>(b);
    // dimension of B
    auto const dimB = GENERATE(take(1, random<std::size_t>(1, 10)));
    // here, instead of b.evaluate(), set value directly
    b.setValue(test::Point{dimB}); // this lets b know about dimB
    // randomize derivative and set wrong dimensions
    b.setDerivative(test::Jacobian{0, 0, 20});

    // expression of function f with random derivative
    auto f               = test::MockOperation<test::Point, test::Jacobian>();
    f.derivative().rows  = dimB;
    f.derivative().cols  = cols;
    f.derivative().value = GENERATE(take(1, random(2, 10)));
    b.setExpression(f);

    auto const resetTangent = GENERATE(false, true);
    if (resetTangent) {
        bBaseRef.setTangentZero(Shape{cols});
        THEN("tangent is zero with correct dimensions")
        {
            CHECK(b.derivative().rows == dimB);
            CHECK(b.derivative().cols == cols);
            CHECK(b.derivative().value == 0);
        }
    }
    WHEN("setting tangent manually")
    {
        // setting derivative just copies
        // no sanity checks on dimensions
        auto const tangent = test::Jacobian{2, 3, 13};
        b.setDerivative(tangent);
        THEN("tangent is retained") { CHECK(b.derivative() == tangent); }
    }
    WHEN("setting tangent to identity")
    {
        b.setDerivativeIdentity();
        THEN("tangent is identity")
        {
            CHECK(b.derivative().rows == dimB);
            CHECK(b.derivative().cols == dimB);
            CHECK(b.derivative().value == 1);
        }
    }
    WHEN("pushing forward tangent by f")
    {
        bBaseRef.pushTangent();
        THEN("tangent is pushed forward to b")
        {
            CHECK(b.derivative() == f.derivative());
        }
    }
}

SCENARIO("Pullback of gradient by f: A -> B", "[Computation]")
{
    // computation of a point b in B
    auto b = Computation<test::Point, test::Jacobian>();
    // some functions must be accessible through the base class
    auto& bBaseRef = dynamic_cast<AbstractComputation&>(b);
    // set a random gradient
    auto const gradient = test::Jacobian{2, 3, 13};
    b.setDerivative(gradient);

    // expression of function f with wrong derivative
    auto f         = test::MockOperation<test::Point, test::Jacobian>();
    f.derivative() = test::Jacobian{0, 0, 2};
    b.setExpression(f);

    WHEN("pulling back gradient by f")
    {
        bBaseRef.pullGradient();
        THEN("gradient is pulled back") { CHECK(f.derivative() == gradient); }
    }
}

SCENARIO("Adding gradient", "[Computation]")
{
    /*
       Note:
       A gradient (covector) is represented by a row vector.
       More generally, we can assign a number of rows to the gradient,
       giving it the shape of a matrix.
       The pullback always has the same number of rows.
     */

    // number of rows in gradient
    auto const rows = GENERATE(take(1, random<std::size_t>(1, 10)));

    // computation of a point b in B
    auto b = Computation<test::Point, test::Jacobian>();
    // some functions must be accessible through the base class
    auto& bBaseRef = dynamic_cast<AbstractComputation&>(b);
    // dimension of B
    auto const dimB = GENERATE(take(1, random<std::size_t>(1, 10)));
    // here, instead of b.evaluate(), set value directly
    // this lets b know about dimB
    b.setValue(test::Point{dimB});
    // for reverse-mode differentiation, are derivatives are reset
    bBaseRef.setGradientZero(Shape{rows});

    THEN("gradient is zero with correct dimensions")
    {
        CHECK(b.derivative().rows == rows);
        CHECK(b.derivative().cols == dimB);
        CHECK(b.derivative().value == 0);
    }
    WHEN("setting gradient manually")
    {
        // setting derivative just copies
        // no sanity checks on dimensions
        auto const gradient = test::Jacobian{2, 3, 13};
        b.setDerivative(gradient);
        THEN("gradient is retained") { CHECK(b.derivative() == gradient); }
    }
    WHEN("setting gradient to identity")
    {
        b.setDerivativeIdentity();
        THEN("gradient is identity")
        {
            CHECK(b.derivative().rows == dimB);
            CHECK(b.derivative().cols == dimB);
            CHECK(b.derivative().value == 1);
        }
    }
    WHEN("adding a gradient")
    {
        auto gradient = test::Jacobian{rows, dimB, 13};
        b.addGradient(gradient);
        THEN("gradient equals added gradient")
        {
            CHECK(b.derivative() == gradient);
        }
        AND_WHEN("adding another gradient")
        {
            auto const gradient2 = test::Jacobian{rows, dimB, -4};
            b.addGradient(gradient2);
            THEN("gradient equals added gradient")
            {
                gradient += gradient2;
                CHECK(b.derivative() == gradient);
            }
        }
    }
}
