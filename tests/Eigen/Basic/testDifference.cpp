#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Difference.hpp>

SCENARIO("x - y", "Basic::Difference")
{
    auto const [pX, pY, value, dX, dY, prec]
        = GENERATE(table({{1.5, 2.0, -0.5, 1.0, -1.0, 1E-6}}));
    CHECK_BINARY_OP(operator-, pX, pY, value, dX, dY, prec);
}
