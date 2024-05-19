// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_POW_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_POW_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X, typename Y>
class Pow : public BinaryOperation<Pow<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Pow<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().pow(Base::yValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& xValue = Base::xValue();
        auto const& yValue = Base::yValue();

        if constexpr (!Base::hasOperandX) {
            return xValue.pow(yValue) * xValue.log() * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return xValue.pow(yValue - 1) * yValue * Base::xPushForward();
        } else {
            return xValue.pow(yValue - 1) * yValue * Base::xPushForward()
                 + xValue.pow(yValue) * xValue.log() * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& xValue = Base::xValue();
        auto const& yValue = Base::yValue();

        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * xValue.pow(yValue - 1) * yValue);
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * xValue.pow(yValue) * xValue.log());
        }
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_ARRAY_OP(pow, EigenAD::Array::Pow)
AUTODIFF_MAKE_ARRAY_SCALAR_OP(pow, EigenAD::Array::Pow)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_POW_HPP
