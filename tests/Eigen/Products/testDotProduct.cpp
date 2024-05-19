#include "common.hpp"

#include <AutoDiff/src/Eigen/Products/ops/DotProduct.hpp>

SCENARIO("dot(x, y)", "EigenAD::DotProduct")
{
    auto const pX    = Eigen::VectorXd{{1.0, 2.0}};
    auto const pY    = Eigen::VectorXd{{-0.5, 1.5}};
    auto const value = 2.5;
    auto const dX    = Eigen::RowVectorXd{{-0.5, 1.5}};
    auto const dY    = Eigen::RowVectorXd{{1.0, 2.0}};
    CHECK_BINARY_OP(dot, pX, pY, value, dX, dY, 1E-6);
}
