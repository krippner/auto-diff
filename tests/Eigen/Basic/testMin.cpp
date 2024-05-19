#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Min.hpp>

SCENARIO("min(x)", "Basic::Min")
{
    auto const [point, value, derivative, prec] = GENERATE(table({
        {-1.5, -1.5, 1.0, 1E-6}, //
        {0.0, 0.0, 0.0, 1E-6},   //
        {1.5, 0.0, 0.0, 1E-6}    //
    }));
    CHECK_UNARY_OP(min, point, value, derivative, prec);
}
