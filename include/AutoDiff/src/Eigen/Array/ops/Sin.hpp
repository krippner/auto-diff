// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_SIN_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_SIN_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Sin : public UnaryOperation<Sin<X>, X> {
public:
    using Base = UnaryOperation<Sin<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().sin();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xValue().cos() * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * Base::xValue().cos());
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(sin, EigenAD::Array::Sin)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_SIN_HPP
