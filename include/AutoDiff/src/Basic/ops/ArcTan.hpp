// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_ARC_TAN_HPP
#define AUTODIFF_SRC_BASIC_OPS_ARC_TAN_HPP

namespace AutoDiff::Basic {

template <typename X>
class ArcTan : public UnaryOperation<ArcTan<X>, X> {
public:
    using Base = UnaryOperation<ArcTan<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::atan(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& x = Base::xValue();
        return 1 / (1 + x * x) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& x = Base::xValue();
        Base::xPullBack(derivative / (1 + x * x));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(atan, Basic::ArcTan)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_ARC_TAN_HPP
