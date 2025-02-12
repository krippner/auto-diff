// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_DIFFERENCE_HPP
#define AUTODIFF_SRC_BASIC_OPS_DIFFERENCE_HPP

namespace AutoDiff::Basic {

template <typename X, typename Y>
class Difference : public Expression<Difference<X, Y>>,
                   public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue() - Op::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Op::hasOperandX) {
            return -Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return Op::xPushForward();
        } else {
            return Op::xPushForward() - Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative);
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(-derivative);
        }
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_BINARY_OP(operator-, Basic::Difference)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_DIFFERENCE_HPP
