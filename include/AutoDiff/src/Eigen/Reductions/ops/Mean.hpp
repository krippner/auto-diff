// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_MEAN_HPP
#define AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_MEAN_HPP

namespace AutoDiff::EigenAD {

template <typename X>
class Mean : public UnaryOperation<Mean<X>, X> {
public:
    using Base = UnaryOperation<Mean<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().mean();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xPushForward().colwise().mean();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const size = Base::xValue().size();
        Base::xPullBack(derivative.replicate(1, size) / size);
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIXBASE_UNARY_OP(mean, EigenAD::Mean)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_MEAN_HPP
