// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_EXP_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_EXP_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Exp : public UnaryOperation<Exp<X>, X> {
public:
    using Base = UnaryOperation<Exp<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().exp();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return _valueImpl() * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * _valueImpl());
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(exp, EigenAD::Array::Exp)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_EXP_HPP
