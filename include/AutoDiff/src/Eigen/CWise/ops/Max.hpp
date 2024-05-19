// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_MAX_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_MAX_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X>
class Max : public UnaryOperation<Max<X>, X> {
public:
    using Base = UnaryOperation<Max<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().cwiseMax(0);
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return xDeriv() * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * xDeriv());
    }

private:
    [[nodiscard]] auto xDeriv() -> decltype(auto)
    {
        return Base::xValue()
            .unaryExpr([](auto x) -> decltype(x) { return x > 0; })
            .reshaped()
            .asDiagonal();
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_UNARY_OP(max, EigenAD::CWise::Max)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_MAX_HPP
