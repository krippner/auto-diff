#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Square.hpp>

SCENARIO("square(x)", "Square")
{
    auto const [point, value, derivative, prec] = GENERATE(table({
        {3.0, 9.0, 6.0, 1E-6},  //
        {-3.0, 9.0, -6.0, 1E-6} //
    }));
    CHECK_UNARY_OP(square, point, value, derivative, prec);
}
