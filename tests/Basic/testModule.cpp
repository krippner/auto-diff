#include <AutoDiff/Basic>
#include <AutoDiff/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using AutoDiff::Boolean;
using AutoDiff::Function;
using AutoDiff::Integer;
using AutoDiff::Real;
using AutoDiff::RealF;
using AutoDiff::var;

using Catch::Matchers::WithinAbsMatcher;

SCENARIO("Integrating Basic module with core classes", "[Basic]")
{
    Real a, b, x, y, z;
    Function f(from(a, b), to(z));
    x = a + b;
    y = a * b;
    z = a / exp(x / y);

    a = 0.5, b = -2.5, f.evaluate();
    CHECK_THAT(z(), WithinAbsMatcher(0.1009483, 1e-6));
    f.pullGradientAt(z);
    CHECK_THAT(d(a), WithinAbsMatcher(0.6056896, 1e-6));
    CHECK_THAT(d(b), WithinAbsMatcher(0.01615172, 1e-6));
}

SCENARIO("Deduced variable types match aliases", "[Basic]")
{
    auto doubleVar = var(0.5);
    CHECK(std::is_same_v<decltype(doubleVar), Real>);

    auto floatVar = var(0.5f);
    CHECK(std::is_same_v<decltype(floatVar), RealF>);

    auto intVar = var(0);
    CHECK(std::is_same_v<decltype(intVar), Integer>);

    auto boolVar = var(true);
    CHECK(std::is_same_v<decltype(boolVar), Boolean>);
}

SCENARIO("Computation with float derivatives", "[Basic]")
{
    RealF x, y, z;
    Function f(from(x, y), to(z));
    z = x * y;

    x = 0.5f, y = -2.5f, f.evaluate();
    CHECK_THAT(z(), WithinAbsMatcher(-1.25f, 1e-6));
    f.pullGradientAt(z);
    CHECK_THAT(d(x), WithinAbsMatcher(-2.5f, 1e-6));
    CHECK_THAT(d(y), WithinAbsMatcher(0.5f, 1e-6));
}
