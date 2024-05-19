#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Product.hpp>

SCENARIO("cwiseProduct(x, y) with x, y in R^(2x2)", "EigenAD::CWise::Product")
{
    auto const pX    = Eigen::MatrixXd{{2.0, 2.0}, {4.0, 2.0}};
    auto const pY    = Eigen::MatrixXd{{1.0, -2.0}, {0.5, 1.5}};
    auto const value = Eigen::MatrixXd{{2.0, -4.0}, {2.0, 3.0}};
    auto const dX    = Eigen::MatrixXd{{1.0, -2.0}, {0.5, 1.5}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    auto const dY = Eigen::MatrixXd{{2.0, 2.0}, {4.0, 2.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    CHECK_BINARY_OP(cwiseProduct, pX, pY, value, dX, dY, 1E-6);
}

SCENARIO("x * y with x in R^(2x2) and y in R", "EigenAD::CWise::ProductScalar")
{
    auto const pX    = Eigen::MatrixXd{{1.5, 1.0}, {2.0, 0.5}};
    auto const pY    = 2.0;
    auto const value = Eigen::MatrixXd{{3.0, 2.0}, {4.0, 1.0}};
    auto const dX    = Eigen::MatrixXd{{2.0, 2.0}, {2.0, 2.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    auto const dY = Eigen::MatrixXd{{1.5, 1.0}, {2.0, 0.5}}.reshaped().eval();
    CHECK_BINARY_OP(operator*, pX, pY, value, dX, dY, 1E-6);
}

SCENARIO(
    "x * y with y in R and x in R^(2x2)", "EigenAD::CWise::ProductScalarMatrix")
{
    auto const pX    = 2.0;
    auto const pY    = Eigen::MatrixXd{{1.5, 1.0}, {2.0, 0.5}};
    auto const value = Eigen::MatrixXd{{3.0, 2.0}, {4.0, 1.0}};
    auto const dX = Eigen::MatrixXd{{1.5, 1.0}, {2.0, 0.5}}.reshaped().eval();
    auto const dY = Eigen::MatrixXd{{2.0, 2.0}, {2.0, 2.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    CHECK_BINARY_OP(operator*, pX, pY, value, dX, dY, 1E-6);
}
