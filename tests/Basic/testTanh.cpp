#include "common.hpp"

#include <AutoDiff/src/Basic/ops/Tanh.hpp>

SCENARIO("tanh(x)", "Tanh")
{
    auto const [point, value, derivative, prec]
        = GENERATE(table({{1.0, 0.7615942, 0.4199743, 1E-6}}));
    CHECK_UNARY_OP(tanh, point, value, derivative, prec);
}
