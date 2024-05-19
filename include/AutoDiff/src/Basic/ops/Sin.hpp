// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_SIN_HPP
#define AUTODIFF_SRC_BASIC_OPS_SIN_HPP

namespace AutoDiff::Basic {

template <typename X>
class Sin : public UnaryOperation<Sin<X>, X> {
public:
    using Base = UnaryOperation<Sin<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::sin(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return std::cos(Base::xValue()) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * std::cos(Base::xValue()));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(sin, Basic::Sin)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_SIN_HPP
