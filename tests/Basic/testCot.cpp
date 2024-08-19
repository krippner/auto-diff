#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Cot.hpp>

SCENARIO("cot(x)", "Cot")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{0.5, 1.830488, -4.350685, 1E-6}}));
    CHECK_UNARY_OP(cot, point, value, derivative, prec);
}
