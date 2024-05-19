// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQRT_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQRT_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Sqrt : public UnaryOperation<Sqrt<X>, X> {
public:
    using Base = UnaryOperation<Sqrt<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().sqrt();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xPushForward() / (2 * Base::xValue().sqrt());
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative / (2 * Base::xValue().sqrt()));
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(sqrt, EigenAD::Array::Sqrt)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQRT_HPP
