// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_TOTAL_HPP
#define AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_TOTAL_HPP

namespace AutoDiff::EigenAD {

template <typename X>
class Total : public Expression<Total<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().sum();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Op::xPushForward().colwise().sum();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const size = Op::xValue().size();
        Op::xPullBack(derivative.replicate(1, size));
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIXBASE_UNARY_OP(total, EigenAD::Total)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_TOTAL_HPP
