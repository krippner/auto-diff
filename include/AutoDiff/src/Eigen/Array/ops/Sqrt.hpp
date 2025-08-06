// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQRT_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQRT_HPP

namespace AutoDiff::EigenAD::ArrayOps {

template <typename X>
class Sqrt : public Expression<Sqrt<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().sqrt();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Op::xPushForward() / (2 * Op::xValue().sqrt());
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative / (2 * Op::xValue().sqrt()));
    }
};

} // namespace AutoDiff::EigenAD::ArrayOps

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(sqrt, EigenAD::ArrayOps::Sqrt)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_SQRT_HPP
