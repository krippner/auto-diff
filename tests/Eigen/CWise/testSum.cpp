#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Sum.hpp>

SCENARIO("x + y with x, y in R^(2x2)", "EigenAD::CWise::Sum")
{
    auto const pX    = Eigen::MatrixXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const pY    = Eigen::MatrixXd{{-1.5, -1.0}, {1.0, 1.5}};
    auto const value = Eigen::MatrixXd{{-2.5, 1.0}, {1.5, 3.0}};
    auto const dX    = Eigen::MatrixXd{{1.0, 1.0}, {1.0, 1.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    auto const dY = Eigen::MatrixXd{{1.0, 1.0}, {1.0, 1.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    CHECK_BINARY_OP(operator+, pX, pY, value, dX, dY, 1E-6);
}

SCENARIO("x + y with x in R^(2x2) and y in R", "EigenAD::CWise::SumScalar")
{
    auto const pX    = Eigen::MatrixXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const pY    = 1.5;
    auto const value = Eigen::MatrixXd{{0.5, 3.5}, {2.0, 3.0}};
    auto const dX    = Eigen::MatrixXd{{1.0, 1.0}, {1.0, 1.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    auto const dY = Eigen::MatrixXd{{1.0, 1.0}, {1.0, 1.0}}.reshaped().eval();
    CHECK_BINARY_OP(operator+, pX, pY, value, dX, dY, 1E-6);
}

SCENARIO(
    "x + y with y in R and x in R^(2x2)", "EigenAD::CWise::SumScalarMatrix")
{
    auto const pX    = 1.5;
    auto const pY    = Eigen::MatrixXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const value = Eigen::MatrixXd{{0.5, 3.5}, {2.0, 3.0}};
    auto const dX = Eigen::MatrixXd{{1.0, 1.0}, {1.0, 1.0}}.reshaped().eval();
    auto const dY = Eigen::MatrixXd{{1.0, 1.0}, {1.0, 1.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    CHECK_BINARY_OP(operator+, pX, pY, value, dX, dY, 1E-6);
}
