// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQUARE_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQUARE_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Square : public UnaryOperation<Square<X>, X> {
public:
    using Base = UnaryOperation<Square<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().square();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return 2 * Base::xValue() * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * 2 * Base::xValue());
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(square, EigenAD::Array::Square)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQUARE_HPP
