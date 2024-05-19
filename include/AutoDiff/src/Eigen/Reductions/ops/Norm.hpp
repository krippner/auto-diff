// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_NORM_HPP
#define AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_NORM_HPP

namespace AutoDiff::EigenAD {

template <typename X>
class Norm : public UnaryOperation<Norm<X>, X> {
public:
    using Base = UnaryOperation<Norm<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().norm();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xValue().reshaped().transpose() / _valueImpl()
             * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(
            derivative * Base::xValue().reshaped().transpose() / _valueImpl());
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIXBASE_UNARY_OP(norm, EigenAD::Norm)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_REDUCTIONS_OPS_NORM_HPP
