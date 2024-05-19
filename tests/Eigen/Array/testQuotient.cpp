#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Quotient.hpp>

SCENARIO("x / y", "EigenAD::Array::Quotient")
{
    auto const pX    = Eigen::ArrayXXd{{2.0, 2.0}, {4.0, 3.0}};
    auto const pY    = Eigen::ArrayXXd{{1.0, -2.0}, {0.5, 1.5}};
    auto const value = Eigen::ArrayXXd{{2.0, -1.0}, {8.0, 2.0}};
    auto const dX    = Eigen::ArrayXXd{{1.0, -0.5}, {2.0, 0.6666667}};
    auto const dY    = Eigen::ArrayXXd{{-2.0, -0.5}, {-16.0, -1.3333333}};
    CHECK_BINARY_OP(operator/, pX, pY, value, dX, dY, 1E-6);
}
