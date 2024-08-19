#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Tan.hpp>

SCENARIO("tan(x)", "Tan")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{1.0, 1.557408, 3.425519, 1E-6}}));
    CHECK_UNARY_OP(tan, point, value, derivative, prec);
}
