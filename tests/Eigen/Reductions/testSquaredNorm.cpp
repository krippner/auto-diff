#include "common.hpp"

#include <AutoDiff/src/Eigen/Reductions/ops/SquaredNorm.hpp>

SCENARIO("squaredNorm(x)", "EigenAD::SquaredNorm")
{
    auto const point      = Eigen::MatrixXd{{1.0, 2.0}, {-0.5, 1.5}};
    auto const value      = 7.5;
    auto const derivative = Eigen::MatrixXd{{2.0, 4.0}, {-1.0, 3.0}}
                                .reshaped()
                                .transpose()
                                .eval();
    CHECK_UNARY_OP(squaredNorm, point, value, derivative, 1E-6);
}
