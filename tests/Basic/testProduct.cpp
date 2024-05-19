#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Product.hpp>

SCENARIO("x * y", "Product")
{
    auto const [pX, pY, value, dX, dY, prec]
        = GENERATE(table({{1.5, 2.5, 3.75, 2.5, 1.5, 1E-6}}));
    CHECK_BINARY_OP(operator*, pX, pY, value, dX, dY, prec);
}
