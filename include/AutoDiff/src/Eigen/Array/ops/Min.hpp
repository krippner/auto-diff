// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_MIN_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_MIN_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Min : public UnaryOperation<Min<X>, X> {
public:
    using Base = UnaryOperation<Min<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().min(0);
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
        return Base::xValue().unaryExpr([](auto x) ->
            typename Base::Derivative::Scalar { return (x < 0) ? 1 : 0; });
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(min, EigenAD::Array::Min)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_MIN_HPP
