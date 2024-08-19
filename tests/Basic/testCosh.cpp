#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Cosh.hpp>

SCENARIO("cosh(x)", "Cosh")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{1.0, 1.543081, 1.175201, 1E-6}}));
    CHECK_UNARY_OP(cosh, point, value, derivative, prec);
}
