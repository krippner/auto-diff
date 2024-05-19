// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_COS_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_COS_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Cos : public UnaryOperation<Cos<X>, X> {
public:
    using Base = UnaryOperation<Cos<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().cos();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return (-Base::xValue()).sin() * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * (-Base::xValue()).sin());
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(cos, EigenAD::Array::Cos)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_COS_HPP
