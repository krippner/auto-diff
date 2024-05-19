#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Quotient.hpp>

SCENARIO("cwiseQuotient(x, y) with x, y in R^(2x2)", "EigenAD::CWise::Quotient")
{
    auto const pX    = Eigen::MatrixXd{{2.0, 2.0}, {4.0, 3.0}};
    auto const pY    = Eigen::MatrixXd{{1.0, -2.0}, {0.5, 1.5}};
    auto const value = Eigen::MatrixXd{{2.0, -1.0}, {8.0, 2.0}};
    auto const dX    = Eigen::MatrixXd{{1.0, -0.5}, {2.0, 0.6666667}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    auto const dY = Eigen::MatrixXd{{-2.0, -0.5}, {-16.0, -1.3333333}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    CHECK_BINARY_OP(cwiseQuotient, pX, pY, value, dX, dY, 1E-6);
}

SCENARIO("x / y with x in R^(2x2) and y in R", "EigenAD::CWise::QuotientScalar")
{
    auto const pX    = Eigen::MatrixXd{{1.5, 1.0}, {2.0, 0.5}};
    auto const pY    = 2.0;
    auto const value = Eigen::MatrixXd{{0.75, 0.5}, {1.0, 0.25}};
    auto const dX    = Eigen::MatrixXd{{0.5, 0.5}, {0.5, 0.5}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    auto const dY
        = Eigen::MatrixXd{{-0.375, -0.25}, {-0.5, -0.125}}.reshaped().eval();
    CHECK_BINARY_OP(operator/, pX, pY, value, dX, dY, 1E-6);
}

SCENARIO("x / y with x in R and y in R^(2x2)",
    "EigenAD::CWise::QuotientScalarMatrix")
{
    auto const pX    = 2.0;
    auto const pY    = Eigen::MatrixXd{{-0.25, 1.0}, {2.0, 0.5}};
    auto const value = Eigen::MatrixXd{{-8.0, 2.0}, {1.0, 4.0}};
    auto const dX = Eigen::MatrixXd{{-4.0, 1.0}, {0.5, 2.0}}.reshaped().eval();
    auto const dY = Eigen::MatrixXd{{-32.0, -2.0}, {-0.5, -8.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    CHECK_BINARY_OP(operator/, pX, pY, value, dX, dY, 1E-6);
}
