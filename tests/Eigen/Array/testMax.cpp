#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Max.hpp>

SCENARIO("max(0, x)", "EigenAD::Array::Max")
{
    auto const point      = Eigen::ArrayXXd{{-1.0, 2.0}, {0.0, 1.5}};
    auto const value      = Eigen::ArrayXXd{{0.0, 2.0}, {0.0, 1.5}};
    auto const derivative = Eigen::ArrayXXd{{0.0, 1.0}, {0.0, 1.0}};
    CHECK_UNARY_OP(max, point, value, derivative, 1E-6);
}
