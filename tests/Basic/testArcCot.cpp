#include "common.hpp"

#include <AutoDiff/src/Basic/ops/ArcCot.hpp>

SCENARIO("acot(x)", "ArcCot")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{0.5, 1.107149, -0.8, 1E-6}}));
    CHECK_UNARY_OP(acot, point, value, derivative, prec);
}
