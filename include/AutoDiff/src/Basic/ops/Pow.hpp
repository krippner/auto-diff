// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_POW_HPP
#define AUTODIFF_SRC_BASIC_OPS_POW_HPP

namespace AutoDiff::Basic {

template <typename X, typename Y>
class Pow : public BinaryOperation<Pow<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Pow<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return std::pow(Base::xValue(), Base::yValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& xValue = Base::xValue();
        auto const& yValue = Base::yValue();

        if constexpr (!Base::hasOperandX) {
            auto const value = std::pow(xValue, yValue);
            return value * std::log(xValue) * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return std::pow(xValue, yValue - 1) * yValue * Base::xPushForward();
        } else {
            auto const value = std::pow(xValue, yValue);
            return std::pow(xValue, yValue - 1) * yValue * Base::xPushForward()
                 + value * std::log(xValue) * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& xValue = Base::xValue();
        auto const& yValue = Base::yValue();

        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * std::pow(xValue, yValue - 1) * yValue);
        }
        if constexpr (Base::hasOperandY) {
            auto const value = std::pow(xValue, yValue);
            Base::yPullBack(derivative * value * std::log(xValue));
        }
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_BINARY_OP(pow, Basic::Pow)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_POW_HPP
