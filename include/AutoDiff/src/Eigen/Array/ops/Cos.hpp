// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_COS_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_COS_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Cos : public Expression<Cos<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().cos();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return (-Op::xValue()).sin() * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * (-Op::xValue()).sin());
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(cos, EigenAD::Array::Cos)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_COS_HPP
