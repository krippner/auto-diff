#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Cos.hpp>

SCENARIO("cos(x)", "EigenAD::CWise::Cos")
{
    auto const point = Eigen::MatrixXd{{-1.0, 2.0}, {0.5, 1.5}};
    auto const value
        = Eigen::MatrixXd{{0.5403023, -0.4161468}, {0.8775826, 0.07073720}};
    auto const derivative
        = Eigen::MatrixXd{{0.8414710, -0.9092974}, {-0.4794255, -0.9974950}}
              .reshaped()
              .asDiagonal()
              .toDenseMatrix();
    CHECK_UNARY_OP(cos, point, value, derivative, 1E-6);
}
