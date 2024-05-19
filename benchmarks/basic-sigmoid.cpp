#include <AutoDiff/Basic>
#include <AutoDiff/Core>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_chronometer.hpp>
#include <catch2/catch_test_macros.hpp>

using AutoDiff::Function;
using AutoDiff::var;

TEST_CASE("Benchmark sigmoid", "[benchmark]")
{
    BENCHMARK_ADVANCED("Building expression")
    (Catch::Benchmark::Chronometer meter)
    {
        meter.measure([] {
            auto x = var(0.5);
            auto k = var(4.0);
            auto z = var(1 / (1 + exp(-k * x)));
            return z;
        });
    };
    BENCHMARK_ADVANCED("Compiling function")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(0.5);
        auto k = var(4.0);
        auto z = var(1 / (1 + exp(-k * x)));

        meter.measure([&] {
            auto f = Function(z);
            f.compile();
            return f;
        });
    };
    BENCHMARK_ADVANCED("Evaluating compiled function")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(0.5);
        auto k = var(4.0);
        auto z = var(1 / (1 + exp(-k * x)));
        auto f = Function(z);
        f.compile();

        meter.measure([&] {
            f.evaluate();
            return f;
        });
    };
    BENCHMARK_ADVANCED("Pushing forward by evaluated function")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(0.5);
        auto k = var(4.0);
        auto z = var(1 / (1 + exp(-k * x)));
        auto f = Function(z);
        f.evaluate();

        meter.measure([&] { f.pushTangentAt(x); });
    };
    BENCHMARK_ADVANCED("Pulling back by evaluated function")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(0.5);
        auto k = var(4.0);
        auto z = var(1 / (1 + exp(-k * x)));
        auto f = Function(z);
        f.evaluate();

        meter.measure([&] { f.pullGradientAt(z); });
    };
}
