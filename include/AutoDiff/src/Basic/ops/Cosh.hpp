// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_COSH_HPP
#define AUTODIFF_SRC_BASIC_OPS_COSH_HPP

namespace AutoDiff::Basic {

template <typename X>
class Cosh : public UnaryOperation<Cosh<X>, X> {
public:
    using Base = UnaryOperation<Cosh<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::cosh(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return std::sinh(Base::xValue()) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * std::sinh(Base::xValue()));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(cosh, Basic::Cosh)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_COSH_HPP
