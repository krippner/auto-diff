#include "common.hpp"

#include <AutoDiff/src/Eigen/CWise/ops/Pow.hpp>

SCENARIO("pow(x, y) with x, y in R^(2x2)", "EigenAD::CWise::Pow")
{
    auto const pX    = Eigen::MatrixXd{{2.0, 2.0}, {4.0, 2.0}};
    auto const pY    = Eigen::MatrixXd{{1.0, 2.0}, {0.5, 1.5}};
    auto const value = Eigen::MatrixXd{{2.0, 4.0}, {2.0, 2.8284271}};
    auto const dX    = Eigen::MatrixXd{{1.0, 4.0}, {0.25, 2.1213203}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    auto const dY
        = Eigen::MatrixXd{{1.3862944, 2.7725887}, {2.7725887, 1.9605163}}
              .reshaped()
              .asDiagonal()
              .toDenseMatrix();
    CHECK_BINARY_OP(pow, pX, pY, value, dX, dY, 1E-6);
}

SCENARIO("pow(x, y) with x in R^(2x2) and y in R", "EigenAD::CWise::Pow")
{
    auto const pX    = Eigen::MatrixXd{{1.5, 1.0}, {2.0, 0.5}};
    auto const pY    = 2.0;
    auto const value = Eigen::MatrixXd{{2.25, 1.0}, {4.0, 0.25}};
    auto const dX    = Eigen::MatrixXd{{3.0, 2.0}, {4.0, 1.0}}
                        .reshaped()
                        .asDiagonal()
                        .toDenseMatrix();
    auto const dY = Eigen::MatrixXd{{0.9122965, 0.0}, {2.7725887, -0.1732868}}
                        .reshaped()
                        .eval();
    CHECK_BINARY_OP(pow, pX, pY, value, dX, dY, 1E-6);
}
