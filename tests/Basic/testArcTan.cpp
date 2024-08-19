#include "common.hpp"

#include <AutoDiff/src/Basic/ops/ArcTan.hpp>

SCENARIO("atan(x)", "ArcTan")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{0.5, 0.4636476, 0.8, 1E-6}}));
    CHECK_UNARY_OP(atan, point, value, derivative, prec);
}
