// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_SIN_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_SIN_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Sin : public Expression<Sin<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().sin();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Op::xValue().cos() * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * Op::xValue().cos());
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(sin, EigenAD::Array::Sin)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_SIN_HPP
