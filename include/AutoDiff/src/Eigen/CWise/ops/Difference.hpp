// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_DIFFERENCE_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_DIFFERENCE_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X, typename Y>
class Difference : public Expression<Difference<X, Y>>,
                   public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue() - Op::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Op::hasOperandX) {
            return -Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return Op::xPushForward();
        } else {
            return Op::xPushForward() - Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative);
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(-derivative);
        }
    }
};

template <typename X, typename Y>
class DifferenceScalar : public Expression<DifferenceScalar<X, Y>>,
                         public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return (Op::xValue().array() - Op::yValue()).matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Op::hasOperandX) {
            auto const size = Op::xValue().size();
            return -Op::yPushForward().replicate(size, 1);
        } else if constexpr (!Op::hasOperandY) {
            return Op::xPushForward();
        } else {
            return Op::xPushForward().rowwise() - Op::yPushForward().row(0);
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative);
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(-derivative.rowwise().sum());
        }
    }
};

template <typename X, typename Y>
class DifferenceScalarMatrix : public Expression<DifferenceScalarMatrix<X, Y>>,
                               public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return (Op::xValue() - Op::yValue().array()).matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Op::hasOperandX) {
            return -Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            auto const size = Op::yValue().size();
            return Op::xPushForward().replicate(size, 1);
        } else {
            auto const size = Op::yValue().size();
            return Op::xPushForward().replicate(size, 1) - Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative.rowwise().sum());
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(-derivative);
        }
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_BINARY_OP(operator-, EigenAD::CWise::Difference)
AUTODIFF_MAKE_CWISE_SCALAR_OP(operator-, EigenAD::CWise::DifferenceScalar)
AUTODIFF_MAKE_CWISE_SCALAR_MATRIX_OP(operator-,
    EigenAD::CWise::DifferenceScalarMatrix)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_DIFFERENCE_HPP
