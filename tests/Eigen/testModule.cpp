#include <AutoDiff/Core>
#include <AutoDiff/Eigen>

#include <Eigen/Core>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using AutoDiff::Boolean;
using AutoDiff::Function;
using AutoDiff::Integer;
using AutoDiff::Real;
using AutoDiff::RealF;
using AutoDiff::var;

using Catch::Matchers::WithinAbsMatcher;

SCENARIO("Integrating Eigen Array module with core classes", "[Eigen]")
{
    auto u = var(Eigen::Array2d{-0.5, 1.5});
    auto x = var(-cos(u) / u);
    auto y = var(square(u));
    auto z = var(x * y);

    auto const targetValue = Eigen::Array2d{0.4387913, -0.1061058};
    CAPTURE(z(), targetValue);
    CHECK(z().isApprox(targetValue, 1e-6));

    Function f(from(u), to(z));
    auto const targetDeriv = Eigen::Array2d{-0.6378698, 1.425505};
    f.pullGradientAt(z);
    CAPTURE(d(u), targetDeriv);
    CHECK(d(u).isApprox(targetDeriv, 1e-6));
}

SCENARIO("Integrating Eigen Matrix module with core classes", "[Eigen]")
{
    auto x  = var(Eigen::Vector2d{0.5, 1.2});
    auto y  = var(Eigen::Vector2d{-2.5, 1.0});
    auto xy = cwiseQuotient(x, y);
    auto z  = var(Eigen::MatrixXd{{-1.0, 2.0}, {0.5, 0.5}, {5.0, 0.0}} * xy);
    auto w  = var(norm(z) + 1);

    CHECK_THAT(w(), WithinAbsMatcher(3.830194, 1e-6));

    Function f(from(x, y), to(w));
    auto const targetDerivX = Eigen::RowVector2d{1.038798, 1.925663};
    auto const targetDerivY = Eigen::RowVector2d{0.2077596, -2.310795};
    f.pullGradientAt(w);
    CAPTURE(d(x), targetDerivX);
    CHECK(d(x).isApprox(targetDerivX, 1e-6));
    CAPTURE(d(y), targetDerivY);
    CHECK(d(y).isApprox(targetDerivY, 1e-6));
}

SCENARIO("Deduced variable types match aliases", "[Eigen]")
{
    auto doubleVar = var(0.5);
    CHECK(std::is_same_v<decltype(doubleVar), Real>);

    auto floatVar = var(0.5f);
    CHECK(std::is_same_v<decltype(floatVar), RealF>);

    auto intVar = var(0);
    CHECK(std::is_same_v<decltype(intVar), Integer>);

    auto boolVar = var(true);
    CHECK(std::is_same_v<decltype(boolVar), Boolean>);

    auto vector2dVar = var(Eigen::Vector2d{0.5, 1.2});
    CHECK(std::is_same_v<decltype(vector2dVar), AutoDiff::Vector2d>);

    auto vectorVar = var(Eigen::VectorXd{});
    CHECK(std::is_same_v<decltype(vectorVar), AutoDiff::Vector>);

    auto vector2fVar = var(Eigen::Vector2f{0.5f, 1.2f});
    CHECK(std::is_same_v<decltype(vector2fVar), AutoDiff::Vector2f>);

    auto vectorXfVar = var(Eigen::VectorXf{});
    CHECK(std::is_same_v<decltype(vectorXfVar), AutoDiff::VectorXf>);

    auto arrayVar = var(Eigen::ArrayXd{});
    CHECK(std::is_same_v<decltype(arrayVar), AutoDiff::Array>);

    auto arrayXfVar = var(Eigen::ArrayXf{});
    CHECK(std::is_same_v<decltype(arrayXfVar), AutoDiff::ArrayXf>);
}

SCENARIO("Computation with float derivatives", "[Eigen]")
{
    auto x  = var(Eigen::Vector2f{0.5f, 1.2f});
    auto y  = var(Eigen::Vector2f{-2.5f, 1.0f});
    auto xy = cwiseQuotient(x, y);
    auto z
        = var(Eigen::MatrixXf{{-1.0f, 2.0f}, {0.5f, 0.5f}, {5.0f, 0.0f}} * xy);
    auto w = var(norm(z) + 1);

    CHECK_THAT(w(), WithinAbsMatcher(3.830194f, 1e-6));

    Function f(from(x, y), to(w));
    auto const targetDerivX = Eigen::RowVector2f{1.038798f, 1.925663f};
    auto const targetDerivY = Eigen::RowVector2f{0.2077596f, -2.310795f};
    f.pullGradientAt(w);
    CAPTURE(d(x), targetDerivX);
    CHECK(d(x).isApprox(targetDerivX, 1e-6));
    CAPTURE(d(y), targetDerivY);
    CHECK(d(y).isApprox(targetDerivY, 1e-6));
}

template <typename T>
using Evaluated_t = AutoDiff::internal::Evaluated_t<T>;

TEST_CASE("Evaluated type trait", "[Eigen]")
{
    STATIC_CHECK(std::is_same_v<Evaluated_t<int>, int>);
    STATIC_CHECK(std::is_same_v<Evaluated_t<float>, float>);
    STATIC_CHECK(std::is_same_v<Evaluated_t<double>, double>);
    STATIC_CHECK(std::is_same_v<Evaluated_t<Eigen::MatrixXd>, Eigen::MatrixXd>);
    STATIC_CHECK(std::is_same_v<
        Evaluated_t<Eigen::Product<Eigen::MatrixXd, Eigen::MatrixXd>>,
        Eigen::MatrixXd>);
}

template <typename T>
using DefaultDerivative_t = AutoDiff::internal::DefaultDerivative_t<T>;

TEST_CASE("DefaultDerivative type trait", "[Eigen]")
{
    SECTION("Types with MatrixXd derivative")
    {
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<int>, Eigen::MatrixXd>);
        STATIC_CHECK(
            std::is_same_v<DefaultDerivative_t<double>, Eigen::MatrixXd>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::MatrixXd>,
            Eigen::MatrixXd>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::VectorXd>,
            Eigen::MatrixXd>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::VectorXi>,
            Eigen::MatrixXd>);
    }
    SECTION("Types with MatrixXf derivative")
    {
        STATIC_CHECK(
            std::is_same_v<DefaultDerivative_t<float>, Eigen::MatrixXf>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::MatrixXf>,
            Eigen::MatrixXf>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::VectorXf>,
            Eigen::MatrixXf>);
    }
    SECTION("Types with Array derivatives")
    {
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::ArrayXi>,
            Eigen::ArrayXd>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::ArrayXXi>,
            Eigen::ArrayXXd>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::ArrayXf>,
            Eigen::ArrayXf>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::ArrayXXf>,
            Eigen::ArrayXXf>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::ArrayXd>,
            Eigen::ArrayXd>);
        STATIC_CHECK(std::is_same_v<DefaultDerivative_t<Eigen::ArrayXXd>,
            Eigen::ArrayXXd>);
    }
}
