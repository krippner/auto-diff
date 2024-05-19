#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Sum.hpp>

SCENARIO("x + y", "Sum")
{
    auto const [pX, pY, value, dX, dY, prec]
        = GENERATE(table({{1.5, 2.5, 4.0, 1.0, 1.0, 1E-6}}));
    CHECK_BINARY_OP(operator+, pX, pY, value, dX, dY, prec);
}
