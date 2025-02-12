// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_MEAN_HPP
#define AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_MEAN_HPP

namespace AutoDiff::EigenAD {

template <typename X>
class Mean : public Expression<Mean<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().mean();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Op::xPushForward().colwise().mean();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const size = Op::xValue().size();
        Op::xPullBack(derivative.replicate(1, size) / size);
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIXBASE_UNARY_OP(mean, EigenAD::Mean)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_MEAN_HPP
