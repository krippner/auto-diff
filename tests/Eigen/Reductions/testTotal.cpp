#include "common.hpp"

#include <AutoDiff/src/Eigen/Reductions/ops/Total.hpp>

SCENARIO("total(x)", "EigenAD::Total")
{
    auto const point = Eigen::MatrixXd{{1.0, 2.0}, {-0.5, 1.5}};
    auto const value = 4.0;
    auto const derivative
        = Eigen::MatrixXd{{1.0, 1.0}, {1.0, 1.0}}.reshaped().transpose().eval();
    CHECK_UNARY_OP(total, point, value, derivative, 1E-6);
}
