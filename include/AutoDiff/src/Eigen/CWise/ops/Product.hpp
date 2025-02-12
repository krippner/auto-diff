// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_PRODUCT_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X, typename Y>
class Product : public Expression<Product<X, Y>>, public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().cwiseProduct(Op::yValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Op::hasOperandX) {
            return yDeriv() * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return xDeriv() * Op::xPushForward();
        } else {
            return xDeriv() * Op::xPushForward()
                 + yDeriv() * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * xDeriv());
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * yDeriv());
        }
    }

private:
    [[nodiscard]] auto xDeriv() -> decltype(auto)
    {
        return Op::yValue().reshaped().asDiagonal();
    }

    [[nodiscard]] auto yDeriv() -> decltype(auto)
    {
        return Op::xValue().reshaped().asDiagonal();
    }
};

template <typename X, typename Y>
class ProductScalar : public Expression<ProductScalar<X, Y>>,
                      public BinaryOperation<X, Y> {
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
            return Op::xValue().reshaped() * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return Op::yValue() * Op::xPushForward();
        } else {
            return Op::yValue() * Op::xPushForward()
                 + Op::xValue().reshaped() * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * Op::yValue());
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * Op::xValue().reshaped());
        }
    }
};

template <typename X, typename Y>
class ProductScalarMatrix : public Expression<ProductScalarMatrix<X, Y>>,
                            public BinaryOperation<X, Y> {
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
            return Op::yValue().reshaped() * Op::xPushForward();
        } else {
            return Op::yValue().reshaped() * Op::xPushForward()
                 + Op::xValue() * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * Op::yValue().reshaped());
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * Op::xValue());
        }
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_BINARY_OP(cwiseProduct, EigenAD::CWise::Product)
AUTODIFF_MAKE_CWISE_SCALAR_OP(operator*, EigenAD::CWise::ProductScalar)
AUTODIFF_MAKE_CWISE_SCALAR_MATRIX_OP(operator*,
    EigenAD::CWise::ProductScalarMatrix)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_PRODUCT_HPP
