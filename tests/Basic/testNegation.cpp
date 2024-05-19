#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Negation.hpp>

SCENARIO("-x", "Negation")
{
    auto const [point, value, derivative, prec] = GENERATE(table({
        {-1.5, 1.5, -1.0, 1E-6}, //
        {1.5, -1.5, -1.0, 1E-6}  //
    }));
    CHECK_UNARY_OP(operator-, point, value, derivative, prec);
}
