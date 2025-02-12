// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_TANH_HPP
#define AUTODIFF_SRC_BASIC_OPS_TANH_HPP

namespace AutoDiff::Basic {

template <typename X>
class Tanh : public Expression<Tanh<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::tanh(Op::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& tanh_x = std::tanh(Op::xValue());
        return (1 - tanh_x * tanh_x) * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& tanh_x = std::tanh(Op::xValue());
        Op::xPullBack(derivative * (1 - tanh_x * tanh_x));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(tanh, Basic::Tanh)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_TANH_HPP
