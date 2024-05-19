// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_PRODUCT_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X, typename Y>
class Product : public BinaryOperation<Product<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Product<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().cwiseProduct(Base::yValue());
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Base::hasOperandX) {
            return yDeriv() * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return xDeriv() * Base::xPushForward();
        } else {
            return xDeriv() * Base::xPushForward()
                 + yDeriv() * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * xDeriv());
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * yDeriv());
        }
    }

private:
    [[nodiscard]] auto xDeriv() -> decltype(auto)
    {
        return Base::yValue().reshaped().asDiagonal();
    }

    [[nodiscard]] auto yDeriv() -> decltype(auto)
    {
        return Base::xValue().reshaped().asDiagonal();
    }
};

template <typename X, typename Y>
class ProductScalar : public BinaryOperation<ProductScalar<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<ProductScalar<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue() * Base::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Base::hasOperandX) {
            return Base::xValue().reshaped() * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return Base::yValue() * Base::xPushForward();
        } else {
            return Base::yValue() * Base::xPushForward()
                 + Base::xValue().reshaped() * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * Base::yValue());
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * Base::xValue().reshaped());
        }
    }
};

template <typename X, typename Y>
class ProductScalarMatrix
    : public BinaryOperation<ProductScalarMatrix<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<ProductScalarMatrix<X, Y>, X, Y>;
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
            return Base::yValue().reshaped() * Base::xPushForward();
        } else {
            return Base::yValue().reshaped() * Base::xPushForward()
                 + Base::xValue() * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * Base::yValue().reshaped());
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * Base::xValue());
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
