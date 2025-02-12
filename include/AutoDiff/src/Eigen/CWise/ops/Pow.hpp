// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_POW_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_POW_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X, typename Y>
class Pow : public Expression<Pow<X, Y>>, public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().array().pow(Op::yValue().array()).matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const xArray = Op::xValue().array();
        auto const yArray = Op::yValue().array();
        if constexpr (!Op::hasOperandX) {
            return yDeriv(xArray, yArray) * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return xDeriv(xArray, yArray) * Op::xPushForward();
        } else {
            return xDeriv(xArray, yArray) * Op::xPushForward()
                 + yDeriv(xArray, yArray) * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const xArray = Op::xValue().array();
        auto const yArray = Op::yValue().array();
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * xDeriv(xArray, yArray));
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * yDeriv(xArray, yArray));
        }
    }

private:
    template <typename XArray, typename YArray>
    [[nodiscard]] static auto xDeriv(XArray const& xArray, YArray const& yArray)
        -> decltype(auto)
    {
        return (xArray.pow(yArray - 1) * yArray)
            .matrix()
            .reshaped()
            .asDiagonal();
    }

    template <typename XArray, typename YArray>
    [[nodiscard]] static auto yDeriv(XArray const& xArray, YArray const& yArray)
        -> decltype(auto)
    {
        return (xArray.pow(yArray) * xArray.log())
            .matrix()
            .reshaped()
            .asDiagonal();
    }
};

/**
 * @brief Special case of Pow<X, Y> for Y = Scalar
 */
template <typename X, typename Y>
class PowScalar : public Expression<PowScalar<X, Y>>,
                  public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue().array().pow(Op::yValue()).matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const xArray = Op::xValue().array();
        auto const yValue = Op::yValue();
        if constexpr (!Op::hasOperandX) {
            return yDeriv(xArray, yValue) * Op::yPushForward();
        } else if constexpr (!Op::hasOperandY) {
            return xDeriv(xArray, yValue) * Op::xPushForward();
        } else {
            return xDeriv(xArray, yValue) * Op::xPushForward()
                 + yDeriv(xArray, yValue) * Op::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const xArray = Op::xValue().array();
        auto const yValue = Op::yValue();
        if constexpr (Op::hasOperandX) {
            Op::xPullBack(derivative * xDeriv(xArray, yValue));
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * yDeriv(xArray, yValue));
        }
    }

private:
    template <typename XArray, typename YValueType>
    [[nodiscard]] static auto xDeriv(
        XArray const& xArray, YValueType const& yValue) -> decltype(auto)
    {
        return (xArray.pow(yValue - 1) * yValue)
            .matrix()
            .reshaped()
            .asDiagonal();
    }

    template <typename XArray, typename YValueType>
    [[nodiscard]] static auto yDeriv(
        XArray const& xArray, YValueType const& yValue) -> decltype(auto)
    {
        return (xArray.pow(yValue) * xArray.log()).matrix().reshaped();
    }
};

} // namespace AutoDiff::EigenAD::CWise

namespace AutoDiff {

AUTODIFF_MAKE_CWISE_BINARY_OP(pow, EigenAD::CWise::Pow)
AUTODIFF_MAKE_CWISE_SCALAR_OP(pow, EigenAD::CWise::PowScalar)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_CWISE_OPS_POW_HPP
