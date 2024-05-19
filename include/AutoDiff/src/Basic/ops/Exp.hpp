// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_EXP_HPP
#define AUTODIFF_SRC_BASIC_OPS_EXP_HPP

namespace AutoDiff::Basic {

template <typename X>
class Exp : public UnaryOperation<Exp<X>, X> {
public:
    using Base = UnaryOperation<Exp<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::exp(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return std::exp(Base::xValue()) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * std::exp(Base::xValue()));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(exp, Basic::Exp)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_EXP_HPP
