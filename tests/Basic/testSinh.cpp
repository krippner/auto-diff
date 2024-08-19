#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Sinh.hpp>

SCENARIO("sinh(x)", "Sinh")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{1.0, 1.175201, 1.543081, 1E-6}}));
    CHECK_UNARY_OP(sinh, point, value, derivative, prec);
}
