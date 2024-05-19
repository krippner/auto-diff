#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Max.hpp>

SCENARIO("max(x)", "Basic::Max")
{
    auto const [point, value, derivative, prec] = GENERATE(table({
        {-1.5, 0.0, 0.0, 1E-6}, //
        {0.0, 0.0, 0.0, 1E-6},  //
        {1.5, 1.5, 1.0, 1E-6}   //
    }));
    CHECK_UNARY_OP(max, point, value, derivative, prec);
}
