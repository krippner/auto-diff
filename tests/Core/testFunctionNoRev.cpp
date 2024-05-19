#include "../helper/int.hpp"

#define AUTODIFF_NO_REVERSE_MODE

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

SCENARIO("A variable as function (no rev)", "[Function]")
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
            AND_WHEN("pushing tangent vector at x")
            {
                f.pushTangentAt(x);
                CHECK(d(x) == derivative);
            }
            AND_WHEN("pushing custom tangent vector at x")
            {
                x.setDerivative(1);
                f.pushTangent();
                CHECK(d(x) == derivative);
            }
        }
    }
}

SCENARIO("Product function (no rev)", "[Function]")
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
            AND_WHEN("pushing tangent vector at x")
            {
                f.pushTangentAt(x);
                CHECK(d(z) == dz_dx);
            }
            AND_WHEN("pushing tangent vector at y")
            {
                f.pushTangentAt(y);
                CHECK(d(z) == dz_dy);
            }
            AND_WHEN("attempting to push tangent vector at z")
            {
                CHECK_THROWS_AS(f.pushTangentAt(z), SeedError);
            }
            AND_WHEN("pushing custom tangent vector at x")
            {
                x.setDerivative(1);
                y.setDerivative(0);
                f.pushTangent();
                CHECK(d(z) == dz_dx);
            }
            AND_WHEN("pushing custom tangent vector at y")
            {
                x.setDerivative(0);
                y.setDerivative(1);
                f.pushTangent();
                CHECK(d(z) == dz_dy);
            }
        }
    }
}

SCENARIO("Recompiling after variable assignment (no rev)", "[Function]")
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
                AND_WHEN("pushing tangent vector at u")
                {
                    f.pushTangentAt(u);
                    CHECK(d(z) == dz_du);
                }
                AND_WHEN("pushing tangent vector at v")
                {
                    f.pushTangentAt(v);
                    CHECK(d(z) == dz_dv);
                }
            }
        }
    }
}

SCENARIO("Function with cyclic dependency (no rev)", "[Function]")
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

SCENARIO("Function with multiple targets (no rev)", "[Function]")
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
            AND_WHEN("pushing tangent vector at x")
            {
                f.pushTangentAt(x);
                CHECK(d(x) == 1); // ∂x/∂x
                CHECK(d(y) == 0); // ∂y/∂x
                CHECK(d(u) == du_dx);
                CHECK(d(v) == dv_dx);
            }
            AND_WHEN("pushing tangent vector at y")
            {
                f.pushTangentAt(y);
                CHECK(d(x) == 0); // ∂x/∂y
                CHECK(d(y) == 1); // ∂y/∂y
                CHECK(d(u) == du_dy);
                CHECK(d(v) == dv_dy);
            }
        }
    }
}

SCENARIO("Function composition (no rev)", "[Function]")
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
            WHEN("pushing tangent vector at u")
            {
                f.pushTangentAt(u);
                g.pushTangent();
                CHECK(d(z) == dz_du);
            }
            WHEN("pushing tangent vector at v")
            {
                f.pushTangentAt(v);
                g.pushTangent();
                CHECK(d(z) == dz_dv);
            }
        }
    }
}

SCENARIO("Differentiation after eager evaluation (no rev)", "[Function]")
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

        WHEN("pushing tangent vector at x")
        {
            f.pushTangentAt(x);
            CHECK(d(z) == 3);
        }
        WHEN("pushing tangent vector at y")
        {
            f.pushTangentAt(y);
            CHECK(d(z) == 2);
        }
    }
}
