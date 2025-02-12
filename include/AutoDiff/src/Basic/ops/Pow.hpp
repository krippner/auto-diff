// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_POW_HPP
#define AUTODIFF_SRC_BASIC_OPS_POW_HPP

namespace AutoDiff::Basic {

template <typename X, typename Y>
class Pow : public Expression<Pow<X, Y>>, public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::pow(Op::xValue(), Op::yValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& xVal = Op::xValue();
        auto const& yVal = Op::yValue();
        if constexpr (!Op::hasOperandX) {
            auto const val = std::pow(xVal, yVal);
            return val * std::log(xVal) * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return std::pow(xVal, yVal - 1) * yVal * Op::xPushForward();
        } else {
            auto const val = std::pow(xVal, yVal);
            return std::pow(xVal, yVal - 1) * yVal * Op::xPushForward()
                 + val * std::log(xVal) * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& xVal = Op::xValue();
        auto const& yVal = Op::yValue();

        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * std::pow(xVal, yVal - 1) * yVal);
        }
        if constexpr (Op::hasOperandY) {
            auto const val = std::pow(xVal, yVal);
            Op::yPullBack(derivative * val * std::log(xVal));
        }
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_BINARY_OP(pow, Basic::Pow)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_POW_HPP
