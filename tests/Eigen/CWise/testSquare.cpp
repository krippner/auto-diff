#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Square.hpp>

SCENARIO("square(x)", "EigenAD::CWise::Square")
{
    auto const point      = Eigen::MatrixXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const value      = Eigen::MatrixXd{{1.0, 4.0}, {0.25, 2.25}};
    auto const derivative = Eigen::MatrixXd{{-2.0, 4.0}, {1.0, 3.0}}
                                .reshaped()
                                .asDiagonal()
                                .toDenseMatrix();
    CHECK_UNARY_OP(square, point, value, derivative, 1E-6);
}
