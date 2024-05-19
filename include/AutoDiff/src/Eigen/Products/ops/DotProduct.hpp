// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_PRODUCTS_DOT_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_PRODUCTS_DOT_PRODUCT_HPP

namespace AutoDiff::EigenAD {

template <typename X, typename Y>
class DotProduct : public BinaryOperation<DotProduct<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<DotProduct<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().dot(Base::yValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Base::hasOperandX) {
            return Base::xValue().transpose() * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return Base::yValue().transpose() * Base::xPushForward();
        } else {
            return Base::yValue().transpose() * Base::xPushForward()
                 + Base::xValue().transpose() * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * Base::yValue().transpose());
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * Base::xValue().transpose());
        }
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_COLVECTOR_BINARY_OP(dot, EigenAD::DotProduct)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_PRODUCTS_DOT_PRODUCT_HPP
