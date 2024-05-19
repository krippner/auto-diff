#include "common.hpp"

#include <AutoDiff/src/Eigen/Array/ops/Sqrt.hpp>

SCENARIO("sqrt(x)", "EigenAD::Array::Sqrt")
{
    auto const point = Eigen::ArrayXXd{{1.0, 2.0}, {0.5, 1.5}};
    auto const value = Eigen::ArrayXXd{{1.0, 1.414214}, {0.7071068, 1.224745}};
    auto const derivative
        = Eigen::ArrayXXd{{0.5, 0.3535534}, {0.7071068, 0.4082483}};
    CHECK_UNARY_OP(sqrt, point, value, derivative, 1E-6);
}
