#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Max.hpp>

SCENARIO("max(0, x)", "EigenAD::CWise::Max")
{
    auto const point      = Eigen::MatrixXd{{-1.0, 2.0}, {0.0, 1.5}};
    auto const value      = Eigen::MatrixXd{{0.0, 2.0}, {0.0, 1.5}};
    auto const derivative = Eigen::MatrixXd{{0.0, 1.0}, {0.0, 1.0}}
                                .reshaped()
                                .asDiagonal()
                                .toDenseMatrix();
    CHECK_UNARY_OP(max, point, value, derivative, 1E-6);
}
