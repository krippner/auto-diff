// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_QUOTIENT_HPP
#define AUTODIFF_SRC_BASIC_OPS_QUOTIENT_HPP

namespace AutoDiff::Basic {

template <typename X, typename Y>
class Quotient : public Expression<Quotient<X, Y>>,
                 public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue() / Op::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& yVal = Op::yValue();

        if constexpr (!Op::hasOperandX) {
            return -(Op::xValue() * Op::yPushForward()) / (yVal * yVal);
        } else if constexpr (!Op::hasOperandY) {
            return Op::xPushForward() / yVal;
        } else {
            return (Op::xPushForward()
                       - (Op::xValue() / yVal) * Op::yPushForward())
                 / yVal;
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& yVal = Op::yValue();

        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative / yVal);
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * (-Op::xValue() / (yVal * yVal)));
        }
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_BINARY_OP(operator/, Basic::Quotient)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_QUOTIENT_HPP
