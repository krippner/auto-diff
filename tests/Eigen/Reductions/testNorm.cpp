#include "common.hpp"

#include <AutoDiff/src/Eigen/Reductions/ops/Norm.hpp>

SCENARIO("norm(x)", "EigenAD::Norm")
{
    auto const point      = Eigen::MatrixXd{{1.0, 2.0}, {-0.5, 1.5}};
    auto const value      = 2.738613;
    auto const derivative = //
        Eigen::MatrixXd{{0.3651484, 0.7302967}, {-0.1825742, 0.5477226}}
            .reshaped()
            .transpose()
            .eval();
    CHECK_UNARY_OP(norm, point, value, derivative, 1E-6);
}
