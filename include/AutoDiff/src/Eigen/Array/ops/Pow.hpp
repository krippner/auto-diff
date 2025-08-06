// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_POW_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_POW_HPP

namespace AutoDiff::EigenAD::ArrayOps {

template <typename X, typename Y>
class Pow : public Expression<Pow<X, Y>>, public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().pow(Op::yValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& xValue = Op::xValue();
        auto const& yValue = Op::yValue();

        if constexpr (!Op::hasOperandX) {
            return xValue.pow(yValue) * xValue.log() * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return xValue.pow(yValue - 1) * yValue * Op::xPushForward();
        } else {
            return xValue.pow(yValue - 1) * yValue * Op::xPushForward()
                 + xValue.pow(yValue) * xValue.log() * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& xValue = Op::xValue();
        auto const& yValue = Op::yValue();

        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * xValue.pow(yValue - 1) * yValue);
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * xValue.pow(yValue) * xValue.log());
        }
    }
};

} // namespace AutoDiff::EigenAD::ArrayOps

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_ARRAY_OP(pow, EigenAD::ArrayOps::Pow)
AUTODIFF_MAKE_ARRAY_SCALAR_OP(pow, EigenAD::ArrayOps::Pow)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_POW_HPP
