#include "common.hpp"

#include <AutoDiff/src/Basic/ops/ArcSin.hpp>

SCENARIO("asin(x)", "ArcSin")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{0.5, 0.5235988, 1.154701, 1E-6}}));
    CHECK_UNARY_OP(asin, point, value, derivative, prec);
}
