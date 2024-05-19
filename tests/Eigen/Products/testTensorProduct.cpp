#include "common.hpp"

#include <AutoDiff/src/Eigen/Products/ops/TensorProduct.hpp>

SCENARIO("tensorProduct(x, y)", "EigenAD::TensorProduct")
{
    auto const pX    = Eigen::VectorXd{{1.0, 2.0}};
    auto const pY    = Eigen::VectorXd{{-0.5, 1.5}};
    auto const value = Eigen::MatrixXd{{-0.5, 1.5}, {-1.0, 3.0}};
    auto const dX
        = Eigen::MatrixXd{{-0.5, 0.0}, {0.0, -0.5}, {1.5, 0.0}, {0.0, 1.5}};
    auto const dY
        = Eigen::MatrixXd{{1.0, 0.0}, {2.0, 0.0}, {0.0, 1.0}, {0.0, 2.0}};
    CHECK_BINARY_OP(tensorProduct, pX, pY, value, dX, dY, 1E-6);
}
