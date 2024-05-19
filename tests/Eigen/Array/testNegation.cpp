#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Negation.hpp>

SCENARIO("-x", "EigenAD::Array::Negation")
{
    auto const point      = Eigen::ArrayXXd{{-1.0, 2.0}, {-0.5, 1.5}};
    auto const value      = Eigen::ArrayXXd{{1.0, -2.0}, {0.5, -1.5}};
    auto const derivative = Eigen::ArrayXXd{{-1.0, -1.0}, {-1.0, -1.0}};
    CHECK_UNARY_OP(operator-, point, value, derivative, 1E-6);
}
