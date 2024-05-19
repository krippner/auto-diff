#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Log.hpp>

SCENARIO("log(x)", "EigenAD::CWise::Log")
{
    auto const point = Eigen::MatrixXd{{1.0, 2.0}, {0.5, 1.5}};
    auto const value
        = Eigen::MatrixXd{{0.0, 0.6931472}, {-0.6931472, 0.4054651}};
    auto const derivative = Eigen::MatrixXd{{1.0, 0.5}, {2.0, 0.6666667}}
                                .reshaped()
                                .asDiagonal()
                                .toDenseMatrix();
    CHECK_UNARY_OP(log, point, value, derivative, 1E-6);
}
