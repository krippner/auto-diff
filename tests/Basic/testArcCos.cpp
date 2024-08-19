#include "common.hpp"

#include <AutoDiff/src/Basic/ops/ArcCos.hpp>

SCENARIO("acos(x)", "ArcCos")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{0.5, 1.047198, -1.154701, 1E-6}}));
    CHECK_UNARY_OP(acos, point, value, derivative, prec);
}
