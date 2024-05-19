// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_CWISE_OPS_POW_HPP
#define AUTODIFF_SRC_EIGEN_CWISE_OPS_POW_HPP

namespace AutoDiff::EigenAD::CWise {

template <typename X, typename Y>
class Pow : public BinaryOperation<Pow<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<Pow<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().array().pow(Base::yValue().array()).matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const xArray = Base::xValue().array();
        auto const yArray = Base::yValue().array();
        if constexpr (!Base::hasOperandX) {
            return yDeriv(xArray, yArray) * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return xDeriv(xArray, yArray) * Base::xPushForward();
        } else {
            return xDeriv(xArray, yArray) * Base::xPushForward()
                 + yDeriv(xArray, yArray) * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const xArray = Base::xValue().array();
        auto const yArray = Base::yValue().array();
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * xDeriv(xArray, yArray));
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * yDeriv(xArray, yArray));
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
class PowScalar : public BinaryOperation<PowScalar<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<PowScalar<X, Y>, X, Y>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().array().pow(Base::yValue()).matrix();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const xArray = Base::xValue().array();
        auto const yValue = Base::yValue();
        if constexpr (!Base::hasOperandX) {
            return yDeriv(xArray, yValue) * Base::yPushForward();
        } else if constexpr (!Base::hasOperandY) {
            return xDeriv(xArray, yValue) * Base::xPushForward();
        } else {
            return xDeriv(xArray, yValue) * Base::xPushForward()
                 + yDeriv(xArray, yValue) * Base::yPushForward();
        }
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        auto const xArray = Base::xValue().array();
        auto const yValue = Base::yValue();
        if constexpr (Base::hasOperandX) {
            Base::xPullBack(derivative * xDeriv(xArray, yValue));
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * yDeriv(xArray, yValue));
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
