#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Pow.hpp>

SCENARIO("pow(x, y)", "Pow")
{
    auto const [pX, pY, value, dX, dY, prec]
        = GENERATE(table({{2.0, 1.5, 2.828427, 2.121320, 1.960516, 1E-6}}));
    CHECK_BINARY_OP(pow, pX, pY, value, dX, dY, prec);
}
