#include <AutoDiff/src/internal/Shape.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_adapters.hpp> // take
#include <catch2/generators/catch_generators_random.hpp>

using AutoDiff::internal::MapDescription;
using AutoDiff::internal::Shape;

using Catch::Generators::random;

SCENARIO("Constructing shape", "[Shape]")
{
    WHEN("passing empty list")
    {
        auto const shape = Shape{};
        CHECK(shape.size() == 0);
        CHECK(shape[0] == 0);
        CHECK(shape[1] == 0);
        CHECK(shape[2] == 0);
        CHECK_THROWS_AS(shape[8], std::out_of_range);
    }
    WHEN("passing 1 integer")
    {
        auto const number = GENERATE(take(1, random<std::size_t>(1, 10)));
        auto const shape  = Shape{number};
        CHECK(shape.size() == 1);
        CHECK(shape[0] == number);
        CHECK(shape[1] == 0);
        CHECK(shape[2] == 0);
        CHECK_THROWS_AS(shape[8], std::out_of_range);
    }
    WHEN("passing 3 integers")
    {
        auto const number0 = GENERATE(take(1, random<std::size_t>(1, 10)));
        auto const number1 = GENERATE(take(1, random<std::size_t>(1, 10)));
        auto const number2 = GENERATE(take(1, random<std::size_t>(1, 10)));
        auto const shape   = Shape{number0, number1, number2};
        CHECK(shape.size() == 3);
        CHECK(shape[0] == number0);
        CHECK(shape[1] == number1);
        CHECK(shape[2] == number2);
        CHECK_THROWS_AS(shape[8], std::out_of_range);
    }
}

SCENARIO("Comparing shapes", "[Shape]")
{
    WHEN("comparing 0d shapes")
    {
        auto const shape = Shape{};
        CHECK(shape == Shape{});
        CHECK_FALSE(shape != Shape{});
    }
    WHEN("comparing 1d shapes")
    {
        auto const number = GENERATE(take(1, random<std::size_t>(1, 10)));
        auto const shape  = Shape{number};
        CHECK(shape == Shape{number});
        CHECK(shape != Shape{number + 1});
        CHECK(shape != Shape{});
        CHECK_FALSE(shape != Shape{number});
        CHECK_FALSE(shape == Shape{number + 1});
        CHECK_FALSE(shape == Shape{});
    }
    WHEN("comparing 2d shapes")
    {
        auto const number0 = GENERATE(take(1, random<std::size_t>(1, 10)));
        auto const number1 = GENERATE(take(1, random<std::size_t>(1, 10)));
        auto const shape   = Shape{number0, number1};
        CHECK(shape == Shape{number0, number1});
        CHECK(shape != Shape{number0 + 1, number1});
        CHECK(shape != Shape{number0});
        CHECK(shape != Shape{});
        CHECK_FALSE(shape != Shape{number0, number1});
        CHECK_FALSE(shape == Shape{number0 + 1, number1});
        CHECK_FALSE(shape == Shape{number0}); // since number1 != 0
        CHECK_FALSE(shape == Shape{});
    }
}

SCENARIO("copying shapes", "[Shape]")
{
    auto const number0 = GENERATE(take(1, random<std::size_t>(1, 10)));
    auto const number1 = GENERATE(take(1, random<std::size_t>(1, 10)));
    auto const shape   = Shape{number0, number1};

    WHEN("copy constructing")
    {
        auto const shapeNew = shape;
        CHECK(shapeNew == shape);
        CHECK(shapeNew.size() == shape.size());
    }
    WHEN("copy assigning")
    {
        auto shapeNew = Shape{};
        shapeNew      = shape;
        CHECK(shapeNew == shape);
        CHECK(shapeNew.size() == shape.size());
    }
}

SCENARIO("Map description", "[MapDescription]")
{
    WHEN("creating default description")
    {
        auto const descr = MapDescription();
        CHECK(descr.state == MapDescription::evaluated);
        CHECK(descr.domainShape == Shape{});
        CHECK(descr.codomainShape == Shape{});
    }
    WHEN("creating description for evaluated map")
    {
        auto const descr
            = MapDescription{MapDescription::evaluated, {1, 2}, {3}};
        CHECK(descr.state == MapDescription::evaluated);
        CHECK(descr.domainShape == Shape{1, 2});
        CHECK(descr.codomainShape == Shape{3});
    }
}
