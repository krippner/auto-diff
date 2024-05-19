// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_QUOTIENT_HPP
#define AUTODIFF_SRC_BASIC_OPS_QUOTIENT_HPP

namespace AutoDiff::Basic {

template <typename X, typename Y>
class Quotient : public BinaryOperation<Quotient<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Quotient<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue() / Base::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& yValue = Base::yValue();

        if constexpr (!Base::hasOperandX) {
            auto const& xValue = Base::xValue();
            return -xValue * Base::yPushForward() / (yValue * yValue);
        } else if constexpr (!Base::hasOperandY) {
            return Base::xPushForward() / yValue;
        } else {
            auto const& xValue = Base::xValue();
            return (Base::xPushForward()
                       - xValue / yValue * Base::yPushForward())
                 / yValue;
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& yValue = Base::yValue();

        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative / yValue);
        }
        if constexpr (Base::hasOperandY) {
            auto const& xValue = Base::xValue();
            Base::yPullBack(derivative * (-xValue / (yValue * yValue)));
        }
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_BINARY_OP(operator/, Basic::Quotient)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_QUOTIENT_HPP
