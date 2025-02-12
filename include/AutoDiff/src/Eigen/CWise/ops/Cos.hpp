// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_COS_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_COS_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X>
class Cos : public Expression<Cos<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().array().cos().matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return (-Op::xValue()).array().sin().matrix().reshaped().asDiagonal()
             * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(
            derivative
            * (-Op::xValue()).array().sin().matrix().reshaped().asDiagonal());
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_UNARY_OP(cos, EigenAD::CWise::Cos)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_COS_HPP
