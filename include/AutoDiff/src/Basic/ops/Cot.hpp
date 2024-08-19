// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_COT_HPP
#define AUTODIFF_SRC_BASIC_OPS_COT_HPP

namespace AutoDiff::Basic {

template <typename X>
class Cot : public UnaryOperation<Cot<X>, X> {
public:
    using Base = UnaryOperation<Cot<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return 1 / std::tan(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& tan_x = std::tan(Base::xValue());
        return (-1 - 1 / (tan_x * tan_x)) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& tan_x = std::tan(Base::xValue());
        Base::xPullBack(derivative * (-1 - 1 / (tan_x * tan_x)));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(cot, Basic::Cot)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_COT_HPP
