#include "../helper/MockOperation.hpp"
#include "../helper/int.hpp"

#include <AutoDiff/src/internal/Evaluator.hpp>
#include <AutoDiff/src/internal/traits.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp> // take
#include <catch2/generators/catch_generators_random.hpp>

using AutoDiff::internal::AbstractEvaluator;
using AutoDiff::internal::Evaluator;

using MockOperation = test::MockOperation<int, int>;

SCENARIO("Templated base type", "[Evaluator]")
{
    // using Base     = AbstractEvaluator<double, int>;
    // using CustomOp = test::MockOperation<float, int>;
    // CHECK(std::is_base_of_v<Base, Evaluator<CustomOp>>);
}

SCENARIO("Evaluator of an expression", "[Evaluator]")
{
    auto expression = MockOperation();
    auto evaluator  = Evaluator(expression);
    WHEN("evaluating")
    {
        auto const valueIn = GENERATE(take(1, random(1, 10)));
        expression.value() = valueIn;

        int valueOut{0};
        evaluator.evaluateTo(valueOut);

        CHECK(valueOut == valueIn);
    }
    WHEN("pushing forward a tangent")
    {
        auto const tangentIn    = GENERATE(take(1, random(1, 10)));
        expression.derivative() = tangentIn;

        int tangentOut{0};
        evaluator.pushForwardTo(tangentOut);

        CHECK(tangentOut == tangentIn);
    }
    WHEN("pulling back a gradient")
    {
        auto const gradient = GENERATE(take(1, random(1, 10)));
        evaluator.pullBack(gradient);
        CHECK(expression.derivative() == gradient);
    }
}
