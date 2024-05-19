#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Sum.hpp>

SCENARIO("x + y", "EigenAD::Array::Sum")
{
    auto const pX    = Eigen::ArrayXXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const pY    = Eigen::ArrayXXd{{-1.5, -1.0}, {1.0, 1.5}};
    auto const value = Eigen::ArrayXXd{{-2.5, 1.0}, {1.5, 3.0}};
    auto const dX    = Eigen::ArrayXXd{{1.0, 1.0}, {1.0, 1.0}};
    auto const dY    = Eigen::ArrayXXd{{1.0, 1.0}, {1.0, 1.0}};
    CHECK_BINARY_OP(operator+, pX, pY, value, dX, dY, 1E-6);
}
