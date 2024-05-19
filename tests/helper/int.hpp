#ifndef TESTS_HELPER_INT_HPP
#define TESTS_HELPER_INT_HPP

#include <AutoDiff/src/internal/TypeImpl.hpp>
#include <AutoDiff/src/internal/traits.hpp>

namespace AutoDiff::internal {

// using `int` for all types reduces test complexity

template <>
struct Evaluated<int> {
    using type = int;
};

template <>
struct DefaultDerivative<int> {
    using type = int;
};

template <>
struct TypeImpl<int> {
    static auto getShape(int const& /*value*/) -> Shape { return {1}; }

    static auto codomainShape(int const& /*derivative*/) -> Shape
    {
        return {1};
    }

    static void generate(int& derivative, MapDescription descr)
    {
        if (descr.state == MapDescription::zero) {
            derivative = 0;
        } else if (descr.state == MapDescription::identity) {
            derivative = 1;
        }
    }

    static void assign(int& value, int const& other) { value = other; }

    static void addTo(int& value, int const& other) { value += other; }
};

} // namespace AutoDiff::internal

#endif // TESTS_HELPER_INT_HPP
