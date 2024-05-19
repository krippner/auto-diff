#include <AutoDiff/src/internal/traits.hpp>

#include <catch2/catch_test_macros.hpp>

namespace test_traits {

struct Value { };

struct EvaluatedType { };

struct Derivative;

struct DefaultDerivative;

} // namespace test_traits

namespace AutoDiff::internal {

template <>
struct Evaluated<test_traits::Value> {
    using type = test_traits::EvaluatedType;
};

template <>
struct DefaultDerivative<test_traits::EvaluatedType> {
    using type = test_traits::DefaultDerivative;
};

} // namespace AutoDiff::internal

using AutoDiff::internal::DefaultDerivative_t;
using AutoDiff::internal::Evaluated_t;

SCENARIO("Evaluation type trait", "[traits]")
{
    CHECK(std::is_same_v<Evaluated_t<test_traits::Value>,
        test_traits::EvaluatedType>);
}

SCENARIO("Default derivative type associated with int", "[traits]")
{
    CHECK(std::is_same_v<DefaultDerivative_t<test_traits::EvaluatedType>,
        test_traits::DefaultDerivative>);
}
