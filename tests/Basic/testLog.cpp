#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Log.hpp>

SCENARIO("log(x)", "Log")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{2.0, 0.6931472, 0.5, 1E-6}}));
    CHECK_UNARY_OP(log, point, value, derivative, prec);
}
