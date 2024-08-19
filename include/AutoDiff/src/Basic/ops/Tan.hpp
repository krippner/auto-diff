// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_TAN_HPP
#define AUTODIFF_SRC_BASIC_OPS_TAN_HPP

namespace AutoDiff::Basic {

template <typename X>
class Tan : public UnaryOperation<Tan<X>, X> {
public:
    using Base = UnaryOperation<Tan<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::tan(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& tan_x = std::tan(Base::xValue());
        return (1 + tan_x * tan_x) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& tan_x = std::tan(Base::xValue());
        Base::xPullBack(derivative * (1 + tan_x * tan_x));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(tan, Basic::Tan)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_TAN_HPP
