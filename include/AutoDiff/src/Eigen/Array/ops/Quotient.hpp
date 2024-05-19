// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_QUOTIENT_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_QUOTIENT_HPP

namespace AutoDiff::EigenAD::Array {

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
        if constexpr (!Base::hasOperandX) {
            return yDeriv() * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return xDeriv() * Base::xPushForward();
        } else {
            return xDeriv() * Base::xPushForward()
                 + yDeriv() * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * xDeriv());
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * yDeriv());
        }
    }

private:
    [[nodiscard]] auto xDeriv() -> decltype(auto) { return 1 / Base::yValue(); }

    [[nodiscard]] auto yDeriv() -> decltype(auto)
    {
        auto const& yValue = Base::yValue();
        return -Base::xValue() / (yValue * yValue);
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_BINARY_OP(operator/, EigenAD::Array::Quotient)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_QUOTIENT_HPP
