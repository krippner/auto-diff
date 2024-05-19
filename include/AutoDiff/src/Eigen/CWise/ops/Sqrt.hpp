// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_SQRT_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_SQRT_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X>
class Sqrt : public UnaryOperation<Sqrt<X>, X> {
public:
    using Base = UnaryOperation<Sqrt<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().cwiseSqrt();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return xDeriv() * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * xDeriv());
    }

private:
    [[nodiscard]] auto xDeriv() -> decltype(auto)
    {
        return (2 * _valueImpl().reshaped().asDiagonal()).inverse();
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_UNARY_OP(sqrt, EigenAD::CWise::Sqrt)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_SQRT_HPP
