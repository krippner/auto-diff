#include "common.hpp"

#include <AutoDiff/src/Eigen/Products/ops/MatrixVectorProduct.hpp>

SCENARIO("x * y with x in R^(2x2) and y in R^2", "EigenAD::MatrixVectorProduct")
{
    auto const pX    = Eigen::MatrixXd{{1.0, 2.0}, {-0.5, 1.5}};
    auto const pY    = Eigen::VectorXd{{0.1, -2.5}};
    auto const value = Eigen::VectorXd{{-4.9, -3.8}};
    auto const dX
        = Eigen::MatrixXd{{0.1, 0.0, -2.5, 0.0}, {0.0, 0.1, 0.0, -2.5}};
    auto const dY = Eigen::MatrixXd{{1.0, 2.0}, {-0.5, 1.5}};
    CHECK_BINARY_OP(operator*, pX, pY, value, dX, dY, 1E-6);
}
