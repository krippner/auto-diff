// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_QUOTIENT_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_QUOTIENT_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X, typename Y>
class Quotient : public BinaryOperation<Quotient<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Quotient<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().cwiseQuotient(Base::yValue());
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
        return Base::yValue().reshaped().asDiagonal().inverse();
    }

    [[nodiscard]] auto yDeriv() -> decltype(auto)
    {
        return (-Base::xValue().array() / Base::yValue().array().square())
            .matrix()
            .reshaped()
            .asDiagonal();
    }
};

template <typename X, typename Y>
class QuotientScalar : public BinaryOperation<QuotientScalar<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<QuotientScalar<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue() / Base::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& yValue = Base::yValue();

        if constexpr (!Base::hasOperandX) {
            return (-Base::xValue().reshaped()) * Base::yPushForward()
                 / (yValue * yValue);
        } else if constexpr (!Base::hasOperandY) {
            return Base::xPushForward() / yValue;
        } else {
            return Base::xPushForward() / yValue
                 - Base::xValue().reshaped() * Base::yPushForward()
                       / (yValue * yValue);
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& yValue = Base::yValue();
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative / yValue);
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(
                derivative * (-Base::xValue().reshaped()) / (yValue * yValue));
        }
    }
};

template <typename X, typename Y>
class QuotientScalarMatrix
    : public BinaryOperation<QuotientScalarMatrix<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<QuotientScalarMatrix<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return (Base::xValue() / Base::yValue().array()).matrix();
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
        return Base::yValue().cwiseInverse().reshaped();
    }

    [[nodiscard]] auto yDeriv() -> decltype(auto)
    {
        return (-Base::xValue() / Base::yValue().array().square())
            .matrix()
            .reshaped()
            .asDiagonal();
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_BINARY_OP(cwiseQuotient, EigenAD::CWise::Quotient)
AUTODIFF_MAKE_CWISE_SCALAR_OP(operator/, EigenAD::CWise::QuotientScalar)
AUTODIFF_MAKE_CWISE_SCALAR_MATRIX_OP(operator/,
    EigenAD::CWise::QuotientScalarMatrix)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_QUOTIENT_HPP
