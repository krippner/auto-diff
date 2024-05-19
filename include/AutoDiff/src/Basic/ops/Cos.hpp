// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_COS_HPP
#define AUTODIFF_SRC_BASIC_OPS_COS_HPP

namespace AutoDiff::Basic {

template <typename X>
class Cos : public UnaryOperation<Cos<X>, X> {
public:
    using Base = UnaryOperation<Cos<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::cos(Base::xValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return -std::sin(Base::xValue()) * Base::xPushForward();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative * (-std::sin(Base::xValue())));
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(cos, Basic::Cos)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_COS_HPP
