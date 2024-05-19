// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_MAX_HPP
#define AUTODIFF_SRC_BASIC_OPS_MAX_HPP

namespace AutoDiff::Basic {

template <typename X>
class Max : public UnaryOperation<Max<X>, X> {
public:
    using Base = UnaryOperation<Max<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        auto const& xValue = Base::xValue();
        return xValue * (xValue > 0);
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xPushForward() * (Base::xValue() > 0);
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if (Base::xValue() > 0) {
            Base::xPullBack(derivative);
        }
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(max, Basic::Max)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_MAX_HPP
