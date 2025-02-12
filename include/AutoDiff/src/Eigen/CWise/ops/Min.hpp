// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_MIN_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_MIN_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X>
class Min : public Expression<Min<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().cwiseMin(0);
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return xDeriv() * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * xDeriv());
    }

private:
    [[nodiscard]] auto xDeriv() -> decltype(auto)
    {
        return Op::xValue()
            .unaryExpr([](auto x) -> decltype(x) { return x < 0; })
            .reshaped()
            .asDiagonal();
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_UNARY_OP(min, EigenAD::CWise::Min)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_MIN_HPP
