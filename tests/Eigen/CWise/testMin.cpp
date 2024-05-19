#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Min.hpp>

SCENARIO("min(0, x)", "EigenAD::CWise::Min")
{
    auto const point      = Eigen::MatrixXd{{-1.0, 2.0}, {0.0, 1.5}};
    auto const value      = Eigen::MatrixXd{{-1.0, 0.0}, {0.0, 0.0}};
    auto const derivative = Eigen::MatrixXd{{1.0, 0.0}, {0.0, 0.0}}
                                .reshaped()
                                .asDiagonal()
                                .toDenseMatrix();
    CHECK_UNARY_OP(min, point, value, derivative, 1E-6);
}
