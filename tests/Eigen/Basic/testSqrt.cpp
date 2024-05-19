#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Sqrt.hpp>

SCENARIO("sqrt(x)", "Basic::Sqrt")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{2.0, 1.414214, 0.3535534, 1E-6}}));
    CHECK_UNARY_OP(sqrt, point, value, derivative, prec);
}
