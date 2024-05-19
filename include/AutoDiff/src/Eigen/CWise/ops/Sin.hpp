// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_SIN_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_SIN_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X>
class Sin : public UnaryOperation<Sin<X>, X> {
public:
    using Base = UnaryOperation<Sin<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().array().sin().matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xValue().array().cos().matrix().reshaped().asDiagonal()
             * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(
            derivative
            * Base::xValue().array().cos().matrix().reshaped().asDiagonal());
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_UNARY_OP(sin, EigenAD::CWise::Sin)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_SIN_HPP
