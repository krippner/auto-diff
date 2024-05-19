#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Sqrt.hpp>

SCENARIO("sqrt(x)", "EigenAD::CWise::Sqrt")
{
    auto const point = Eigen::MatrixXd{{1.0, 2.0}, {0.5, 1.5}};
    auto const value = Eigen::MatrixXd{{1.0, 1.414214}, {0.7071068, 1.224745}};
    auto const derivative = Eigen::MatrixXd{{0.5, 0.3535534},
        {0.7071068, 0.4082483}}.reshaped()
                                .asDiagonal()
                                .toDenseMatrix();
    CHECK_UNARY_OP(sqrt, point, value, derivative, 1E-6);
}
