// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_NEGATION_HPP
#define AUTODIFF_SRC_BASIC_OPS_NEGATION_HPP

namespace AutoDiff::Basic {

template <typename X>
class Negation : public UnaryOperation<Negation<X>, X> {
public:
    using Base = UnaryOperation<Negation<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return -Base::xValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return -Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(-derivative);
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(operator-, Basic::Negation)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_NEGATION_HPP
