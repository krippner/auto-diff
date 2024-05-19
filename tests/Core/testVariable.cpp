#include "../helper/int.hpp"
#include "helper/Identity.hpp"

#include <AutoDiff/src/Core/Variable.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp> // take
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp> // Equals

using AutoDiff::var;
using test::identity;

using AutoDiff::Variable;
using test::IdentityWithId;

using Catch::Generators::random;
using Catch::Matchers::Equals;

SCENARIO("Constructing integer literal", "[Variable]")
{
    WHEN("using default constructor")
    {
        Variable<int, int> const x;
        CHECK(x() == 0);
    }
    WHEN("using constructor with initial value")
    {
        auto const value = GENERATE(take(1, random(1, 10)));
        Variable<int, int> const x(value);
        CHECK(x() == value);
    }
    WHEN("using factory")
    {
        auto const value = GENERATE(take(1, random(1, 10)));
        auto const x     = var(value);
        CHECK(x() == value);
    }
}

SCENARIO("Copy-constructing and -assigning a variable", "[Variable]")
{
    GIVEN("a literal variable")
    {
        auto const value = GENERATE(take(1, random(1, 10)));
        auto const f     = var(value);
        WHEN("copy constructing a new variable")
        {
            auto const g(f); // NOLINT(*-unnecessary-copy-*)
            CHECK(f == g);
        }
        WHEN("assigning to a new variable")
        {
            Variable<int, int> g;
            g = f;
            CHECK(f == g);
        }
    }
}

SCENARIO("Constructing expression evaluator", "[Variable]")
{
    GIVEN("a literal variable")
    {
        auto const value = GENERATE(take(1, random(1, 10)));
        auto const x     = var(value);
        WHEN("using factory on variable itself")
        {
            auto const y = var(x);
            THEN("new variable is evaluated eagerly") { CHECK(y() == value); }
            THEN("new variable is not a copy") { CHECK(x != y); }
        }
        WHEN("using factory on identity expression")
        {
            auto const y = var(identity(x));
            THEN("new variable is evaluated eagerly") { CHECK(y() == value); }
        }
        WHEN("using constructor on identity expression")
        {
            Variable<int, int> const y{identity(x)};
            THEN("new variable is evaluated eagerly") { CHECK(y() == value); }
        }
    }
}

SCENARIO("Assignment to a literal variable", "[Variable]")
{
    GIVEN("a literal variable")
    {
        auto const value = GENERATE(take(1, random(1, 10)));
        auto x           = var(value);
        WHEN("assigning a new value")
        {
            auto const newValue = value + GENERATE(take(1, random(1, 10)));
            x                   = newValue;
            CHECK(x() == newValue);
        }
        WHEN("assigning an identity expression")
        {
            auto const newValue = GENERATE(take(1, random(1, 10)));
            x                   = identity(var(newValue));
            THEN("variable is eagerly re-evaluated") { CHECK(x() == newValue); }
        }
    }
}

SCENARIO("Assignment to an expression evaluator", "[Variable]")
{
    GIVEN("an expression variable")
    {
        auto x = var(identity(var(0)));
        WHEN("assigning a literal value")
        {
            auto const value = GENERATE(take(1, random(1, 10)));
            x                = value;
            CHECK(x() == value);
        }
        WHEN("assigning another expression")
        {
            auto const value = GENERATE(take(1, random(1, 10)));
            x                = identity(var(value));
            THEN("variable is eagerly re-evaluated") { CHECK(x() == value); }
        }
    }
}

SCENARIO("Accessing the stored derivative", "[Variable]")
{
    GIVEN("a literal variable")
    {
        auto x = var(0);
        THEN("derivative is default constructed") { CHECK(d(x) == 0); }
        WHEN("setting a new derivative value")
        {
            auto const derivative = GENERATE(take(1, random(1, 10)));
            x.setDerivative(derivative);
            CHECK(d(x) == derivative);
        }
    }
}

SCENARIO("Variables are destructed sequentially", "[Variable]")
{
    // Sequential calls of destructors is critical
    // to avoid stack overflow for deep computation graphs
    auto dtorSeq = std::vector<int>();
    {
        auto root = var(0);
        {
            auto const var0 = var(0);
            auto const var1 = var(IdentityWithId{var0, 1, &dtorSeq});
            auto const var2 = var(IdentityWithId{var1, 2, &dtorSeq});
            auto const var3 = var(IdentityWithId{var2, 3, &dtorSeq});
            root            = var(IdentityWithId{var3, 4, &dtorSeq});
        }                // destruct var0, ... , var3
        dtorSeq.clear(); // clear trace of temporaries
    }                    // destruct root
    THEN("child operations are destructed before their parents")
    {
        // Note on dtor order:
        // root dtor is called first, which then initiates the destruction
        // of its successors in a child-before-parents order.
        CHECK_THAT(dtorSeq, Equals<int>({4, 1, 2, 3}));
    }
}
