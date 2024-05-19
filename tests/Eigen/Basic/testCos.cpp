#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Cos.hpp>

SCENARIO("cos(x)", "Basic::Cos")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{2.0, -0.4161468, -0.9092974, 1E-6}}));
    CHECK_UNARY_OP(cos, point, value, derivative, prec);
}
