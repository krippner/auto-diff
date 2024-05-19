#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Pow.hpp>

SCENARIO("pow(x, y)", "EigenAD::Array::Pow")
{
    auto const pX    = Eigen::ArrayXXd{{2.0, 2.0}, {4.0, 2.0}};
    auto const pY    = Eigen::ArrayXXd{{1.0, 2.0}, {0.5, 1.5}};
    auto const value = Eigen::ArrayXXd{{2.0, 4.0}, {2.0, 2.8284271}};
    auto const dX    = Eigen::ArrayXXd{{1.0, 4.0}, {0.25, 2.1213203}};
    auto const dY
        = Eigen::ArrayXXd{{1.3862944, 2.7725887}, {2.7725887, 1.9605163}};
    CHECK_BINARY_OP(pow, pX, pY, value, dX, dY, 1E-6);
}
