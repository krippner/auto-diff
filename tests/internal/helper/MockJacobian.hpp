#ifndef TESTS_INTERNAL_HELPER_MOCK_JACOBIAN_HPP
#define TESTS_INTERNAL_HELPER_MOCK_JACOBIAN_HPP

#include <AutoDiff/src/internal/Shape.hpp>
#include <AutoDiff/src/internal/TypeImpl.hpp>
#include <AutoDiff/src/internal/traits.hpp>

#include <stdexcept>

// define some mock types with shape info ---------------------------

namespace test {

// Mock point on a manifold with dimension dim.
struct Point {
    std::size_t dim{0};
};

// Mock Jacobian matrix.
struct Jacobian {
    std::size_t rows{0};
    std::size_t cols{0};
    int value{0};

    auto operator+=(Jacobian const& other)
    {
        if (rows != other.rows || cols != other.cols) {
            throw std::logic_error{
                "Jacobian matrices have incompatible shapes"};
        }
        value += other.value;
    }
};

auto operator==(Jacobian const& left, Jacobian const& right) -> bool
{
    return left.rows == right.rows && left.cols == right.cols
        && left.value == right.value;
}

} // namespace test

// enable mock types for use in Computation<Point, Jacobian>

namespace AutoDiff::internal {

template <>
struct Evaluated<test::Point> {
    using type = test::Point;
};

template <>
struct TypeImpl<test::Point> {
    static auto getShape(test::Point const& value) -> Shape
    {
        return {value.dim};
    }

    static void assign(test::Point& value, test::Point const& other)
    {
        value = other;
    }
};

template <>
struct TypeImpl<test::Jacobian> {
    static auto codomainShape(test::Jacobian const& derivative) -> Shape
    {
        return {derivative.rows};
    }

    static void generate(
        test::Jacobian& derivative, MapDescription const& descr)
    {
        if (descr.state == MapDescription::zero) {
            derivative.rows  = descr.codomainShape[0];
            derivative.cols  = descr.domainShape[0];
            derivative.value = 0;
        } else if (descr.state == MapDescription::identity) {
            derivative.rows = derivative.cols = descr.domainShape[0];
            derivative.value                  = 1;
        }
    }

    static void assign(test::Point& value, test::Point const& other)
    {
        value = other;
    }

    static void assign(test::Jacobian& derivative, test::Jacobian const& other)
    {
        derivative = other;
    }

    static void addTo(test::Jacobian& derivative, test::Jacobian const& other)
    {
        derivative += other;
    }
};

} // namespace AutoDiff::internal

#endif // TESTS_INTERNAL_HELPER_MOCK_JACOBIAN_HPP
