#include "../helper/int.hpp"

#define AUTODIFF_NO_FORWARD_MODE

#include <AutoDiff/src/Core/Function.hpp>

#include <AutoDiff/src/Core/Variable.hpp>

// these must be included before including individual operations
#include <AutoDiff/src/Basic/factories.hpp>
#include <AutoDiff/src/Core/BinaryOperation.hpp>
#include <AutoDiff/src/Core/UnaryOperation.hpp>
// Basic module must be tested before Core
#include <AutoDiff/src/Basic/ops/Product.hpp>
#include <AutoDiff/src/Basic/ops/Sum.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>          // values
#include <catch2/generators/catch_generators_adapters.hpp> // take

using AutoDiff::CyclicDependencyError;
using AutoDiff::Function;
using AutoDiff::SeedError;
using AutoDiff::var;

using Integer = AutoDiff::Variable<int, int>;

using Catch::Generators::values;

// workaround: template arguments of Catch::Generators::table cannot be deduced

namespace testFunction {

inline auto table(std::initializer_list<std::tuple<int, int>> tuples)
{
    return Catch::Generators::values(tuples);
}

inline auto table(
    std::initializer_list<std::tuple<int, int, int, int, int>> tuples)
{
    return Catch::Generators::values(tuples);
}

inline auto table(
    std::initializer_list<std::tuple<int, int, int, int, int, int, int, int>>
        tuples)
{
    return Catch::Generators::values(tuples);
}

} // namespace testFunction

using testFunction::table;

SCENARIO("A variable as function (no fwd)", "[Function]")
{
    GIVEN("f: Z → Z, x ↦ x")
    {
        Integer x;
        Function f(from(x), to(x));

        // randomize derivatives
        x.setDerivative(-13);

        auto const [value, derivative] = GENERATE(table({{2, 1}}));

        WHEN("compiling f")
        {
            CHECK_FALSE(f.compiled());
            f.compile();
            CHECK(f.compiled());
        }
        WHEN("evaluating at x = value")
        {
            x = value, f.evaluate();
            THEN("f is correctly evaluated")
            {
                CHECK(f.compiled());
                CHECK(x() == value);
            }
            AND_WHEN("pulling gradient at x")
            {
                f.pullGradientAt(x);
                CHECK(d(x) == derivative);
            }
            AND_WHEN("pulling custom gradient at x")
            {
                x.setDerivative(1);
                f.pullGradient();
                CHECK(d(x) == derivative);
            }
        }
    }
}

SCENARIO("Product function (no fwd)", "[Function]")
{
    GIVEN("f: Z ⨉ Z → Z, (x, y) ↦ z = x * y")
    {
        Integer x, y, z;
        Function f(from(x, y), to(z));
        z = x * y;

        // randomize derivatives
        x.setDerivative(-13);
        y.setDerivative(-13);
        z.setDerivative(-13);

        WHEN("compiling f")
        {
            CHECK_FALSE(f.compiled());
            f.compile();
            CHECK(f.compiled());
        }
        WHEN("evaluating at (x, y) = (px, py)")
        {
            auto const [px, py, pz, dz_dx, dz_dy]
                = GENERATE(table({{2, 3, 6, 3, 2}}));
            x = px, y = py, f.evaluate();
            THEN("f is correctly evaluated")
            {
                CHECK(f.compiled());
                CHECK(z() == pz);
            }
            AND_WHEN("pulling gradient")
            {
                f.pullGradientAt(z);
                CHECK(d(x) == dz_dx);
                CHECK(d(y) == dz_dy);
            }
            AND_WHEN("attempting to pull gradient at x")
            {
                CHECK_THROWS_AS(f.pullGradientAt(x), SeedError);
            }
            AND_WHEN("attempting to pull gradient at y")
            {
                CHECK_THROWS_AS(f.pullGradientAt(y), SeedError);
            }
            AND_WHEN("pulling custom gradient")
            {
                z.setDerivative(1);
                f.pullGradient();
                CHECK(d(x) == dz_dx);
                CHECK(d(y) == dz_dy);
            }
        }
    }
}

SCENARIO("Recompiling after variable assignment (no fwd)", "[Function]")
{
    GIVEN("f: Z ⨉ Z → Z, (x, y) ↦ z = x * y")
    {
        Integer x, y, z;
        Function f(from(x, y), to(z));
        z = x * y;

        // randomize derivatives
        x.setDerivative(-13);
        y.setDerivative(-13);
        z.setDerivative(-13);

        f.compile(); // already tested above

        WHEN("assigning new expression to z and recompiling f")
        {
            Integer u, v;
            z = u * v;   // invalidates f
            f.compile(); // f should be valid again
            THEN("f is compiled successfully") { CHECK(f.compiled()); }
            WHEN("evaluating at (u, v) = (pu, pv)")
            {
                auto const [pu, pv, pz, dz_du, dz_dv]
                    = GENERATE(table({{5, 7, 35, 7, 5}}));
                u = pu, v = pv, f.evaluate();
                THEN("f is correctly evaluated") { CHECK(z() == pz); }
                AND_WHEN("pulling gradient")
                {
                    f.pullGradientAt(z);
                    CHECK(d(u) == dz_du);
                    CHECK(d(v) == dz_dv);
                }
            }
        }
    }
}

SCENARIO("Function with cyclic dependency (no fwd)", "[Function]")
{
    GIVEN("f: Z ⨉ Z → Z, (x, y) ↦ x = x * y")
    {
        Integer x, y;
        Function f(to(x));
        // Function f(from(x, y), to(x)) reduces to f = x -> no error
        x = x * y; // introduces cyclic dependency
        WHEN("attempting to compile f")
        {
            CHECK_FALSE(f.compiled());
            CHECK_THROWS_AS(f.compile(), CyclicDependencyError);
            CHECK_FALSE(f.compiled());
        }
        WHEN("attempting to evaluate f")
        {
            CHECK_THROWS_AS(f.evaluate(), CyclicDependencyError);
            CHECK_FALSE(f.compiled());
        }
    }
}

SCENARIO("Function with multiple targets (no fwd)", "[Function]")
{
    GIVEN("f: Z ⨉ Z → Z ⨉ Z, (x, y) ↦ (u, v) = (x + y, x * y)")
    {
        Integer x, y, u, v;
        Function f(from(x, y), to(u, v));
        u = x + y;
        v = x * y;

        // randomize derivatives
        x.setDerivative(-13);
        y.setDerivative(-13);
        u.setDerivative(-13);
        v.setDerivative(-13);

        WHEN("compiling f")
        {
            CHECK_FALSE(f.compiled());
            f.compile();
            CHECK(f.compiled());
        }
        WHEN("evaluating at (x, y) = (px, py)")
        {
            auto const [px, py, pu, pv, du_dx, du_dy, dv_dx, dv_dy]
                = GENERATE(table({{2, 3, 5, 6, 1, 1, 3, 2}}));
            x = px, y = py, f.evaluate();
            THEN("f is correctly evaluated")
            {
                CHECK(f.compiled());
                CHECK(u() == pu);
                CHECK(v() == pv);
            }
            AND_WHEN("pulling gradient at u")
            {
                f.pullGradientAt(u);
                CHECK(d(x) == du_dx);
                CHECK(d(y) == du_dy);
                CHECK(d(u) == 1); // ∂u/∂u
                CHECK(d(v) == 0); // ∂u/∂v
            }
            AND_WHEN("pulling gradient at v")
            {
                f.pullGradientAt(v);
                CHECK(d(x) == dv_dx);
                CHECK(d(y) == dv_dy);
                CHECK(d(u) == 0); // ∂v/∂u
                CHECK(d(v) == 1); // ∂v/∂v
            }
        }
    }
}

SCENARIO("Function composition (no fwd)", "[Function]")
{
    GIVEN("f: (u, v) ↦ (x, y) = (u + v, u * v), g: (x, y) ↦ z = x * y")
    {
        /**
         *      z
         *     / \
         *    x   y
         *   / \ /
         *  u   v
         */

        Integer u, v, x, y;
        Function f(from(u, v), to(x, y));
        x = u + v;
        y = u * v;

        Integer z;
        Function g(from(x, y), to(z));
        z = x * y;

        // randomize derivatives
        x.setDerivative(-13);
        y.setDerivative(-13);
        z.setDerivative(-13);
        u.setDerivative(-13);
        v.setDerivative(-13);

        WHEN("evaluating g after f at (u, v) = (pu, pv)")
        {
            // z = g(f(u, v)) = u²v + uv²
            // ∂z/∂u = 2uv + v²
            // ∂z/∂v = u² + 2uv
            auto const [pu, pv, pz, dz_du, dz_dv]
                = GENERATE(table({{2, 3, 30, 21, 16}}));
            u = pu, v = pv;
            f.evaluate();
            g.evaluate();
            THEN("f,g are correctly evaluated")
            {
                CHECK(f.compiled());
                CHECK(g.compiled());
                CHECK(z() == pz);
            }
            WHEN("pulling gradient at z")
            {
                g.pullGradientAt(z);
                f.pullGradient();
                CHECK(d(u) == dz_du);
                CHECK(d(v) == dz_dv);
            }
        }
    }
}

SCENARIO("Differentiation after eager evaluation (no fwd)", "[Function]")
{
    GIVEN("f: Z ⨉ Z → Z, (x, y) ↦ z")
    {
        auto x = var(2);
        auto y = var(3);
        auto z = var(x * y); // eager evaluation
        Function f(from(x, y), to(z));

        // randomize derivatives
        x.setDerivative(-13);
        y.setDerivative(-13);
        z.setDerivative(-13);

        // no f.evaluate() here

        WHEN("pulling gradient at z")
        {
            f.pullGradientAt(z);
            CHECK(d(x) == 3);
            CHECK(d(y) == 2);
        }
    }
}
