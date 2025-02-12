// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_SINH_HPP
#define AUTODIFF_SRC_BASIC_OPS_SINH_HPP

namespace AutoDiff::Basic {

template <typename X>
class Sinh : public Expression<Sinh<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::sinh(Op::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return std::cosh(Op::xValue()) * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * std::cosh(Op::xValue()));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(sinh, Basic::Sinh)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_SINH_HPP
