// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_EXP_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_EXP_HPP

namespace AutoDiff::EigenAD::ArrayOps {

template <typename X>
class Exp : public Expression<Exp<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().exp();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return _valueImpl() * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * _valueImpl());
    }
};

} // namespace AutoDiff::EigenAD::ArrayOps

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(exp, EigenAD::ArrayOps::Exp)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_EXP_HPP
