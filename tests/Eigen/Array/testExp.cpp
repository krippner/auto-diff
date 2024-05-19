#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Exp.hpp>

SCENARIO("exp(x)", "EigenAD::Array::Exp")
{
    auto const point = Eigen::ArrayXXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const value
        = Eigen::ArrayXXd{{0.3678794, 7.389056}, {1.648721, 4.481689}};
    auto const derivative
        = Eigen::ArrayXXd{{0.3678794, 7.389056}, {1.648721, 4.481689}};
    CHECK_UNARY_OP(exp, point, value, derivative, 1E-6);
}
