#include <AutoDiff/Core>
#include <AutoDiff/Eigen>

#include <Eigen/Core>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_chronometer.hpp>
#include <catch2/catch_test_macros.hpp>

using AutoDiff::Function;
using AutoDiff::var;

TEST_CASE("Benchmark sigmoid", "[benchmark,Eigen]")
{
    BENCHMARK_ADVANCED("Building expression")
    (Catch::Benchmark::Chronometer meter)
    {
        meter.measure([] {
            auto x = var(Eigen::ArrayXd::Constant(1000, 0.5));
            auto k = var(Eigen::ArrayXd::Constant(1000, 4.0));
            auto z = var(1 / (1 + exp(-k * x)));
            return z;
        });
    };
    BENCHMARK_ADVANCED("Compiling function")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(Eigen::ArrayXd::Constant(1000, 0.5));
        auto k = var(Eigen::ArrayXd::Constant(1000, 4.0));
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
        auto x = var(Eigen::ArrayXd::Constant(1000, 0.5));
        auto k = var(Eigen::ArrayXd::Constant(1000, 4.0));
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
        auto x = var(Eigen::ArrayXd::Constant(1000, 0.5));
        auto k = var(Eigen::ArrayXd::Constant(1000, 4.0));
        auto z = var(1 / (1 + exp(-k * x)));
        auto f = Function(z);
        f.evaluate();

        meter.measure([&] { f.pushTangentAt(x); });
    };
    BENCHMARK_ADVANCED("Pushing forward with intermediate variable 2")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(Eigen::ArrayXd::Constant(1000, 0.5));
        auto k = var(Eigen::ArrayXd::Constant(1000, 4.0));
        auto a = var(-k * x);
        auto z = var(1 / (1 + exp(a)));
        auto f = Function(z);
        f.evaluate();

        meter.measure([&] { f.pushTangentAt(x); });
    };
    BENCHMARK_ADVANCED("Pulling back by evaluated function")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(Eigen::ArrayXd::Constant(1000, 0.5));
        auto k = var(Eigen::ArrayXd::Constant(1000, 4.0));
        auto z = var(1 / (1 + exp(-k * x)));
        auto f = Function(z);
        f.evaluate();

        meter.measure([&] { f.pullGradientAt(z); });
    };
    BENCHMARK_ADVANCED("Pulling back with intermediate variable")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(Eigen::ArrayXd::Constant(1000, 0.5));
        auto k = var(Eigen::ArrayXd::Constant(1000, 4.0));
        auto a = var(exp(-k * x));
        auto z = var(1 / (1 + a));
        auto f = Function(z);
        f.evaluate();

        meter.measure([&] { f.pullGradientAt(z); });
    };
    BENCHMARK_ADVANCED("Pulling back with intermediate variable 2")
    (Catch::Benchmark::Chronometer meter)
    {
        auto x = var(Eigen::ArrayXd::Constant(1000, 0.5));
        auto k = var(Eigen::ArrayXd::Constant(1000, 4.0));
        auto a = var(-k * x);
        auto z = var(1 / (1 + exp(a)));
        auto f = Function(z);
        f.evaluate();

        meter.measure([&] { f.pullGradientAt(z); });
    };
}
