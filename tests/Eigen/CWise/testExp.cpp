#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Exp.hpp>

SCENARIO("exp(x)", "EigenAD::CWise::Exp")
{
    auto const point = Eigen::MatrixXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const value
        = Eigen::MatrixXd{{0.3678794, 7.389056}, {1.648721, 4.481689}};
    auto const derivative
        = Eigen::MatrixXd{{0.3678794, 7.389056}, {1.648721, 4.481689}} //
              .reshaped()
              .asDiagonal()
              .toDenseMatrix();
    CHECK_UNARY_OP(exp, point, value, derivative, 1E-6);
}
