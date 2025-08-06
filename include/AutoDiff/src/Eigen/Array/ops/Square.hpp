// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQUARE_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQUARE_HPP

namespace AutoDiff::EigenAD::ArrayOps {

template <typename X>
class Square : public Expression<Square<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().square();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return 2 * Op::xValue() * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * 2 * Op::xValue());
    }
};

} // namespace AutoDiff::EigenAD::ArrayOps

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(square, EigenAD::ArrayOps::Square)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQUARE_HPP
