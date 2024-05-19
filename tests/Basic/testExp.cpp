#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Exp.hpp>

SCENARIO("exp(x)", "Exp")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{2.0, 7.389056, 7.389056, 1E-6}}));
    CHECK_UNARY_OP(exp, point, value, derivative, prec);
}
