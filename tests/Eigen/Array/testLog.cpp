#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Log.hpp>

SCENARIO("log(x)", "EigenAD::Array::Log")
{
    auto const point = Eigen::ArrayXXd{{1.0, 2.0}, {0.5, 1.5}};
    auto const value
        = Eigen::ArrayXXd{{0.0, 0.6931472}, {-0.6931472, 0.4054651}};
    auto const derivative = Eigen::ArrayXXd{{1.0, 0.5}, {2.0, 0.6666667}};
    CHECK_UNARY_OP(log, point, value, derivative, 1E-6);
}
