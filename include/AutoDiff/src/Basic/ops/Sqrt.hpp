// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_SQRT_HPP
#define AUTODIFF_SRC_BASIC_OPS_SQRT_HPP

namespace AutoDiff::Basic {

template <typename X>
class Sqrt : public UnaryOperation<Sqrt<X>, X> {
public:
    using Base = UnaryOperation<Sqrt<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::sqrt(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xPushForward() / (2 * std::sqrt(Base::xValue()));
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative / (2 * std::sqrt(Base::xValue())));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(sqrt, Basic::Sqrt)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_SQRT_HPP
