// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_BASIC_OPS_PRODUCT_HPP
#define AUTODIFF_SRC_BASIC_OPS_PRODUCT_HPP

namespace AutoDiff::Basic {

template <typename X, typename Y>
class Product : public BinaryOperation<Product<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Product<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue() * Base::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Base::hasOperandX) {
            return Base::xValue() * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return Base::yValue() * Base::xPushForward();
        } else {
            return Base::yValue() * Base::xPushForward()
                 + Base::xValue() * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * Base::yValue());
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * Base::xValue());
        }
    }
};

} // namespace AutoDiff::Basic

namespace AutoDiff {

AUTODIFF_MAKE_BASIC_BINARY_OP(operator*, Basic::Product)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_BASIC_OPS_PRODUCT_HPP
