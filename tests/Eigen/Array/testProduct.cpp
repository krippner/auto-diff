#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Product.hpp>

SCENARIO("x * y", "EigenAD::Array::Product")
{
    auto const pX    = Eigen::ArrayXXd{{2.0, 2.0}, {4.0, 2.0}};
    auto const pY    = Eigen::ArrayXXd{{1.0, -2.0}, {0.5, 1.5}};
    auto const value = Eigen::ArrayXXd{{2.0, -4.0}, {2.0, 3.0}};
    auto const dX    = Eigen::ArrayXXd{{1.0, -2.0}, {0.5, 1.5}};
    auto const dY    = Eigen::ArrayXXd{{2.0, 2.0}, {4.0, 2.0}};
    CHECK_BINARY_OP(operator*, pX, pY, value, dX, dY, 1E-6);
}
