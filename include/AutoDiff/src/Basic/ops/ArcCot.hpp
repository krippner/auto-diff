// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_ARC_COT_HPP
#define AUTODIFF_SRC_BASIC_OPS_ARC_COT_HPP

namespace AutoDiff::Basic {

template <typename X>
class ArcCot : public UnaryOperation<ArcCot<X>, X> {
public:
    using Base = UnaryOperation<ArcCot<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::atan(1 / Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& x = Base::xValue();
        return -1 / (1 + x * x) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& x = Base::xValue();
        Base::xPullBack(derivative / -(1 + x * x));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(acot, Basic::ArcCot)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_ARC_COT_HPP
