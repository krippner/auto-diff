// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_SUM_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_SUM_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X, typename Y>
class Sum : public BinaryOperation<Sum<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Sum<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue() + Base::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Base::hasOperandX) {
            return Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return Base::xPushForward();
        } else {
            return Base::xPushForward() + Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative);
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative);
        }
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_BINARY_OP(operator+, EigenAD::Array::Sum);

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_SUM_HPP
