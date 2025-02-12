// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_SQRT_HPP
#define AUTODIFF_SRC_BASIC_OPS_SQRT_HPP

namespace AutoDiff::Basic {

template <typename X>
class Sqrt : public Expression<Sqrt<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::sqrt(Op::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Op::xPushForward() / (2 * std::sqrt(Op::xValue()));
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative / (2 * std::sqrt(Op::xValue())));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(sqrt, Basic::Sqrt)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_SQRT_HPP
