#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Sin.hpp>

SCENARIO("sin(x)", "Basic::Sin")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{2.0, 0.9092974, -0.4161468, 1E-6}}));
    CHECK_UNARY_OP(sin, point, value, derivative, prec);
}
