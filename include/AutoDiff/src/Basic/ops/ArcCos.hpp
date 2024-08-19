// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_ARC_COS_HPP
#define AUTODIFF_SRC_BASIC_OPS_ARC_COS_HPP

namespace AutoDiff::Basic {

template <typename X>
class ArcCos : public UnaryOperation<ArcCos<X>, X> {
public:
    using Base = UnaryOperation<ArcCos<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::acos(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& x = Base::xValue();
        return -std::pow(1 - x * x, -0.5) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& x = Base::xValue();
        Base::xPullBack(derivative * (-std::pow(1 - x * x, -0.5)));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(acos, Basic::ArcCos)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_ARC_COS_HPP
