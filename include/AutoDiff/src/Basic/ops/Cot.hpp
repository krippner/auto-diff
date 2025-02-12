// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_COT_HPP
#define AUTODIFF_SRC_BASIC_OPS_COT_HPP

namespace AutoDiff::Basic {

template <typename X>
class Cot : public Expression<Cot<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return 1 / std::tan(Op::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& tan_x = std::tan(Op::xValue());
        return (-1 - 1 / (tan_x * tan_x)) * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& tan_x = std::tan(Op::xValue());
        Op::xPullBack(derivative * (-1 - 1 / (tan_x * tan_x)));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(cot, Basic::Cot)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_COT_HPP
