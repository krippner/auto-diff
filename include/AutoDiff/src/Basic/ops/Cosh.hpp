// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_COSH_HPP
#define AUTODIFF_SRC_BASIC_OPS_COSH_HPP

namespace AutoDiff::Basic {

template <typename X>
class Cosh : public Expression<Cosh<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::cosh(Op::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return std::sinh(Op::xValue()) * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * std::sinh(Op::xValue()));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(cosh, Basic::Cosh)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_COSH_HPP
