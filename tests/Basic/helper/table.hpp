#ifndef TESTS_BASIC_HELPER_TABLE_HPP
#define TESTS_BASIC_HELPER_TABLE_HPP

#include <catch2/generators/catch_generators.hpp> // values

// workaround for:
// template arguments of Catch::Generators::table cannot be deduced

inline auto table(
    std::initializer_list<std::tuple<double, double, double, double>> tuples)
{
    return Catch::Generators::values(tuples);
}

inline auto table(std::initializer_list<
    std::tuple<double, double, double, double, double, double>>
        tuples)
{
    return Catch::Generators::values(tuples);
}

#endif // TESTS_BASIC_HELPER_TABLE_HPP
