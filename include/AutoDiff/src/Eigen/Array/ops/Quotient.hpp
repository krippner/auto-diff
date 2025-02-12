// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_QUOTIENT_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_QUOTIENT_HPP

namespace AutoDiff::EigenAD::Array {

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
        if constexpr (!Op::hasOperandX) {
            return yDeriv() * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return xDeriv() * Op::xPushForward();
        } else {
            return xDeriv() * Op::xPushForward()
                 + yDeriv() * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * xDeriv());
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * yDeriv());
        }
    }

private:
    [[nodiscard]] auto xDeriv() -> decltype(auto) { return 1 / Op::yValue(); }

    [[nodiscard]] auto yDeriv() -> decltype(auto)
    {
        auto const& yValue = Op::yValue();
        return -Op::xValue() / (yValue * yValue);
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_BINARY_OP(operator/, EigenAD::Array::Quotient)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_QUOTIENT_HPP
