#include <AutoDiff/src/internal/TypeImpl.hpp>

#include <catch2/catch_test_macros.hpp>

using AutoDiff::internal::Shape;

namespace test_type_impl {

struct Value {
    Shape shape{1};
};

struct Derivative {
    int value{0};
};

} // namespace test_type_impl

namespace AutoDiff::internal {

template <>
struct TypeImpl<test_type_impl::Value> {
    static auto getShape(test_type_impl::Value const& value) -> Shape
    {
        return value.shape;
    }
};

template <>
struct TypeImpl<test_type_impl::Derivative> {
    static void generate(
        test_type_impl::Derivative& derivative, MapDescription const& info)
    {
        if (info.state == MapDescription::evaluated) {
            derivative.value = 13;
        } else if (info.state == MapDescription::zero) {
            derivative.value = 0;
        } else if (info.state == MapDescription::identity) {
            derivative.value = 1;
        }
    }

    static void assign(test_type_impl::Derivative& derivative,
        test_type_impl::Derivative const& other)
    {
        derivative.value = other.value;
    }

    static void addTo(test_type_impl::Derivative& derivative,
        test_type_impl::Derivative const& other)
    {
        derivative.value += other.value;
    }
};

} // namespace AutoDiff::internal

using AutoDiff::internal::addTo;
using AutoDiff::internal::assign;
using AutoDiff::internal::generate;
using AutoDiff::internal::getShape;
using AutoDiff::internal::MapDescription;

SCENARIO("Get the shape of an integer", "[TypeImpl]")
{
    test_type_impl::Value value{};
    CHECK(getShape(value) == Shape{1});
}

SCENARIO("Generate derivatives", "[TypeImpl]")
{
    test_type_impl::Derivative derivative{-1};
    WHEN("generating derivative with evaluated state")
    {
        generate(derivative, MapDescription{MapDescription::evaluated, {}, {}});
        CHECK(derivative.value == 13);
    }
    WHEN("generating zero derivative")
    {
        generate(derivative, MapDescription{MapDescription::zero, {}, {}});
        CHECK(derivative.value == 0);
    }
    WHEN("generating identity derivative")
    {
        generate(derivative, MapDescription{MapDescription::identity, {}, {}});
        CHECK(derivative.value == 1);
    }
}

SCENARIO("Assign derivatives", "[TypeImpl]")
{
    test_type_impl::Derivative derivative1{1};
    test_type_impl::Derivative derivative2{2};
    WHEN("assigning derivative2 to derivative1")
    {
        assign(derivative1, derivative2);
        CHECK(derivative1.value == 2);
    }
    WHEN("adding derivative2 to derivative1")
    {
        addTo(derivative1, derivative2);
        CHECK(derivative1.value == 3);
    }
}
