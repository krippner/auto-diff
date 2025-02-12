// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_SQUARE_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_SQUARE_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X>
class Square : public Expression<Square<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        auto const& xValue = Op::xValue();
        return xValue.cwiseProduct(xValue);
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return 2 * Op::xValue().reshaped().asDiagonal() * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * 2 * Op::xValue().reshaped().asDiagonal());
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_UNARY_OP(square, EigenAD::CWise::Square)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_SQUARE_HPP
