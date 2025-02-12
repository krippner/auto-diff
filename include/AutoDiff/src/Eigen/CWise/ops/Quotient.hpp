// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_QUOTIENT_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_QUOTIENT_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X, typename Y>
class Quotient : public Expression<Quotient<X, Y>>,
                 public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().cwiseQuotient(Op::yValue());
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
        return Op::yValue().reshaped().asDiagonal().inverse();
    }

    [[nodiscard]] auto yDeriv() -> decltype(auto)
    {
        return (-Op::xValue().array() / Op::yValue().array().square())
            .matrix()
            .reshaped()
            .asDiagonal();
    }
};

template <typename X, typename Y>
class QuotientScalar : public Expression<QuotientScalar<X, Y>>,
                       public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue() / Op::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& yValue = Op::yValue();

        if constexpr (!Op::hasOperandX) {
            return (-Op::xValue().reshaped()) * Op::yPushForward()
                 / (yValue * yValue);
        } else if constexpr (!Op::hasOperandY) {
            return Op::xPushForward() / yValue;
        } else {
            return Op::xPushForward() / yValue
                 - Op::xValue().reshaped() * Op::yPushForward()
                       / (yValue * yValue);
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const& yValue = Op::yValue();
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative / yValue);
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(
                derivative * (-Op::xValue().reshaped()) / (yValue * yValue));
        }
    }
};

template <typename X, typename Y>
class QuotientScalarMatrix : public Expression<QuotientScalarMatrix<X, Y>>,
                             public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return (Op::xValue() / Op::yValue().array()).matrix();
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
        return Op::yValue().cwiseInverse().reshaped();
    }

    [[nodiscard]] auto yDeriv() -> decltype(auto)
    {
        return (-Op::xValue() / Op::yValue().array().square())
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
