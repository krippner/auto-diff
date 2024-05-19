#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Square.hpp>

SCENARIO("square(x)", "EigenAD::Array::Square")
{
    auto const point      = Eigen::ArrayXXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const value      = Eigen::ArrayXXd{{1.0, 4.0}, {0.25, 2.25}};
    auto const derivative = Eigen::ArrayXXd{{-2.0, 4.0}, {1.0, 3.0}};
    CHECK_UNARY_OP(square, point, value, derivative, 1E-6);
}
