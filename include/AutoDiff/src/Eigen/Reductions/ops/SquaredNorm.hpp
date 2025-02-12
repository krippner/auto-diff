// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_SQUARED_NORM_HPP
#define AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_SQUARED_NORM_HPP

namespace AutoDiff::EigenAD {

template <typename X>
class SquaredNorm : public Expression<SquaredNorm<X>>,
                    public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().squaredNorm();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Op::xValue().reshaped().transpose() * 2 * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * Op::xValue().reshaped().transpose() * 2);
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIXBASE_UNARY_OP(squaredNorm, EigenAD::SquaredNorm)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_SQUARED_NORM_HPP
