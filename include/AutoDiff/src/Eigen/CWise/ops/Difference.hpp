// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_DIFFERENCE_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_DIFFERENCE_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X, typename Y>
class Difference : public BinaryOperation<Difference<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Difference<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue() - Base::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Base::hasOperandX) {
            return -Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return Base::xPushForward();
        } else {
            return Base::xPushForward() - Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative);
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(-derivative);
        }
    }
};

template <typename X, typename Y>
class DifferenceScalar : public BinaryOperation<DifferenceScalar<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<DifferenceScalar<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return (Base::xValue().array() - Base::yValue()).matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Base::hasOperandX) {
            auto const size = Base::xValue().size();
            return -Base::yPushForward().replicate(size, 1);
        } else if constexpr (!Base::hasOperandY) {
            return Base::xPushForward();
        } else {
            return Base::xPushForward().rowwise() - Base::yPushForward().row(0);
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative);
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(-derivative.rowwise().sum());
        }
    }
};

template <typename X, typename Y>
class DifferenceScalarMatrix
    : public BinaryOperation<DifferenceScalarMatrix<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<DifferenceScalarMatrix<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return (Base::xValue() - Base::yValue().array()).matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        if constexpr (!Base::hasOperandX) {
            return -Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            auto const size = Base::yValue().size();
            return Base::xPushForward().replicate(size, 1);
        } else {
            auto const size = Base::yValue().size();
            return Base::xPushForward().replicate(size, 1)
                 - Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative.rowwise().sum());
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(-derivative);
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
