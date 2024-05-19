#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Quotient.hpp>

SCENARIO("x / y", "Quotient")
{
    auto const [pX, pY, value, dX, dY, prec]
        = GENERATE(table({{2.0, 4.0, 0.5, 0.25, -0.125, 1E-6}}));
    CHECK_BINARY_OP(operator/, pX, pY, value, dX, dY, prec);
}
