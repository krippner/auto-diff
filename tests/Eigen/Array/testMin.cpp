#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Min.hpp>

SCENARIO("min(0, x)", "EigenAD::Array::Min")
{
    auto const point      = Eigen::ArrayXXd{{-1.0, 2.0}, {0.0, 1.5}};
    auto const value      = Eigen::ArrayXXd{{-1.0, 0.0}, {0.0, 0.0}};
    auto const derivative = Eigen::ArrayXXd{{1.0, 0.0}, {0.0, 0.0}};
    CHECK_UNARY_OP(min, point, value, derivative, 1E-6);
}
