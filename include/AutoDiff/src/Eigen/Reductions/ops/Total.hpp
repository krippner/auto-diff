// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_TOTAL_HPP
#define AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_TOTAL_HPP

namespace AutoDiff::EigenAD {

template <typename X>
class Total : public UnaryOperation<Total<X>, X> {
public:
    using Base = UnaryOperation<Total<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().sum();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xPushForward().colwise().sum();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const size = Base::xValue().size();
        Base::xPullBack(derivative.replicate(1, size));
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIXBASE_UNARY_OP(total, EigenAD::Total)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_TOTAL_HPP
