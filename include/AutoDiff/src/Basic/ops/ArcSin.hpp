// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_ARC_SIN_HPP
#define AUTODIFF_SRC_BASIC_OPS_ARC_SIN_HPP

namespace AutoDiff::Basic {

template <typename X>
class ArcSin : public Expression<ArcSin<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::asin(Op::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& x = Op::xValue();
        return std::pow(1 - x * x, -0.5) * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& x = Op::xValue();
        Op::xPullBack(derivative * std::pow(1 - x * x, -0.5));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(asin, Basic::ArcSin)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_ARC_SIN_HPP
