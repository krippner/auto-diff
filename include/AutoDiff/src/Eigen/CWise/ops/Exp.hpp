// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_EXP_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_EXP_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X>
class Exp : public Expression<Exp<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().array().exp().matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return _valueImpl().reshaped().asDiagonal() * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * _valueImpl().reshaped().asDiagonal());
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_UNARY_OP(exp, EigenAD::CWise::Exp)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_EXP_HPP
