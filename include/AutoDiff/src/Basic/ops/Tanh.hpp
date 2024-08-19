// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_TANH_HPP
#define AUTODIFF_SRC_BASIC_OPS_TANH_HPP

namespace AutoDiff::Basic {

template <typename X>
class Tanh : public UnaryOperation<Tanh<X>, X> {
public:
    using Base = UnaryOperation<Tanh<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::tanh(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& tanh_x = std::tanh(Base::xValue());
        return (1 - tanh_x * tanh_x) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& tanh_x = std::tanh(Base::xValue());
        Base::xPullBack(derivative * (1 - tanh_x * tanh_x));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(tanh, Basic::Tanh)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_TANH_HPP
