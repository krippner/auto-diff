#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Negation.hpp>

SCENARIO("-x", "EigenAD::CWise::Negation")
{
    auto const point      = Eigen::MatrixXd{{-1.0, 2.0}, {-0.5, 1.5}};
    auto const value      = Eigen::MatrixXd{{1.0, -2.0}, {0.5, -1.5}};
    auto const derivative = Eigen::MatrixXd{{-1.0, -1.0}, {-1.0, -1.0}}
                                .reshaped()
                                .asDiagonal()
                                .toDenseMatrix();
    CHECK_UNARY_OP(operator-, point, value, derivative, 1E-6);
}
