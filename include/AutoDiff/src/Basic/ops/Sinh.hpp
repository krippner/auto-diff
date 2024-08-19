// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_SINH_HPP
#define AUTODIFF_SRC_BASIC_OPS_SINH_HPP

namespace AutoDiff::Basic {

template <typename X>
class Sinh : public UnaryOperation<Sinh<X>, X> {
public:
    using Base = UnaryOperation<Sinh<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::sinh(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return std::cosh(Base::xValue()) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * std::cosh(Base::xValue()));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(sinh, Basic::Sinh)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_SINH_HPP
