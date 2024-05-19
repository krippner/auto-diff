#include "common.hpp"

#include <AutoDiff/src/Eigen/Products/ops/MatrixProduct.hpp>

SCENARIO("x * y with x, y in R^(2x2)", "EigenAD::MatrixProduct")
{
    auto const pX    = Eigen::MatrixXd{{1.0, 2.0}, {-0.5, 1.5}};
    auto const pY    = Eigen::MatrixXd{{0.1, -2.5}, {3.0, -4.0}};
    auto const value = Eigen::MatrixXd{{6.1, -10.5}, {4.45, -4.75}};
    auto const dX    = Eigen::MatrixXd{//
        {0.1, 0.0, 3.0, 0.0},       //
        {0.0, 0.1, 0.0, 3.0},       //
        {-2.5, 0.0, -4.0, 0.0},     //
        {0.0, -2.5, 0.0, -4.0}};
    auto const dY    = Eigen::MatrixXd{//
        {1.0, 2.0, 0.0, 0.0},       //
        {-0.5, 1.5, 0.0, 0.0},      //
        {0.0, 0.0, 1.0, 2.0},       //
        {0.0, 0.0, -0.5, 1.5}};
    CHECK_BINARY_OP(operator*, pX, pY, value, dX, dY, 1E-6);
}
