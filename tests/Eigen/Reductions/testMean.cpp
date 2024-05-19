#include "common.hpp"

#include <AutoDiff/src/Eigen/Reductions/ops/Mean.hpp>

SCENARIO("mean(x)", "EigenAD::Mean")
{
    auto const point      = Eigen::MatrixXd{{1.0, 2.0}, {-0.5, 1.5}};
    auto const value      = 1.0;
    auto const derivative = Eigen::MatrixXd{{0.25, 0.25}, {0.25, 0.25}}
                                .reshaped()
                                .transpose()
                                .eval();
    CHECK_UNARY_OP(mean, point, value, derivative, 1E-6);
}
