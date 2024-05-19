// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_SQUARE_HPP
#define AUTODIFF_SRC_BASIC_OPS_SQUARE_HPP

namespace AutoDiff::Basic {

template <typename X>
class Square : public UnaryOperation<Square<X>, X> {
public:
    using Base = UnaryOperation<Square<X>, X>;
    using Base::Base;
    using Base::operator=;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        auto const& value = Base::xValue();
        return value * value;
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xValue() * 2 * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * Base::xValue() * 2);
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(square, Basic::Square)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_SQUARE_HPP
