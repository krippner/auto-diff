// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_MAX_HPP
#define AUTODIFF_SRC_BASIC_OPS_MAX_HPP

namespace AutoDiff::Basic {

template <typename X>
class Max : public Expression<Max<X>>, public UnaryOperation<X> {
public:
    using Op = UnaryOperation<X>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        auto const& val = Op::xValue();
        return val * (val > 0);
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Op::xPushForward() * (Op::xValue() > 0);
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if (Op::xValue() > 0) {
            Op::xPullBack(derivative);
        }
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_UNARY_OP(max, Basic::Max)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_MAX_HPP
