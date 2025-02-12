// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_SQUARE_HPP
#define AUTODIFF_SRC_BASIC_OPS_SQUARE_HPP

namespace AutoDiff::Basic {

template <typename X>
class Square : public Expression<Square<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;
    using Op::operator=;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        auto const& value = Op::xValue();
        return value * value;
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Op::xValue() * 2 * Op::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Op::xPullBack(derivative * Op::xValue() * 2);
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(square, Basic::Square)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_SQUARE_HPP
