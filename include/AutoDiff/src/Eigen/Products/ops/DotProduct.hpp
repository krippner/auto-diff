// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_PRODUCTS_DOT_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_PRODUCTS_DOT_PRODUCT_HPP

namespace AutoDiff::EigenAD {

template <typename X, typename Y>
class DotProduct : public Expression<DotProduct<X, Y>>,
                   public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().dot(Op::yValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Op::hasOperandX) {
            return Op::xValue().transpose() * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return Op::yValue().transpose() * Op::xPushForward();
        } else {
            return Op::yValue().transpose() * Op::xPushForward()
                 + Op::xValue().transpose() * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * Op::yValue().transpose());
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * Op::xValue().transpose());
        }
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_COLVECTOR_BINARY_OP(dot, EigenAD::DotProduct)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_PRODUCTS_DOT_PRODUCT_HPP
