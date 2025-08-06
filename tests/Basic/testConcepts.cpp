#include <AutoDiff/src/Basic/concepts.hpp>
#include <AutoDiff/src/Core/Variable.hpp>

#include <catch2/catch_test_macros.hpp>

template <typename T>
concept Scalar = AutoDiff::Basic::Scalar<T>;

TEST_CASE("Basic::Scalar concept", "[Basic]")
{
    STATIC_CHECK(Scalar<bool>);
    STATIC_CHECK(Scalar<int>);
    STATIC_CHECK(Scalar<float>);
    STATIC_CHECK(Scalar<double>);
}

template <typename T>
concept ExpressionType = AutoDiff::Basic::ExpressionType<T>;

template <typename Value, typename Derivative>
using Variable = AutoDiff::Variable<Value, Derivative>;

TEST_CASE("Basic::ExpressionType concept", "[Basic]")
{
    STATIC_CHECK(ExpressionType<Variable<double, double>>);
}
