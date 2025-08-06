// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_PRODUCT_HPP

namespace AutoDiff::EigenAD::ArrayOps {

template <typename X, typename Y>
class Product : public Expression<Product<X, Y>>, public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue() * Op::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Op::hasOperandX) {
            return Op::xValue() * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return Op::yValue() * Op::xPushForward();
        } else {
            return Op::yValue() * Op::xPushForward()
                 + Op::xValue() * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * Op::yValue());
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * Op::xValue());
        }
    }
};

} // namespace AutoDiff::EigenAD::ArrayOps

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_BINARY_OP(operator*, EigenAD::ArrayOps::Product)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_PRODUCT_HPP
