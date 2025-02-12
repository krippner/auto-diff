// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_TENSOR_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_TENSOR_PRODUCT_HPP

namespace AutoDiff::EigenAD {

template <typename X, typename Y>
class TensorProduct : public Expression<TensorProduct<X, Y>>,
                      public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;
    using typename Op::Derivative;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue() * Op::yValue().transpose();
    }

    [[nodiscard]] auto _pushForwardImpl() -> Derivative
    {
        auto const& xValue = Op::xValue();
        auto const& yValue = Op::yValue();
        Derivative deriv;

        if constexpr (!Op::hasOperandX) {
            auto const& yDerivative = Op::yPushForward();
            auto const derivCols    = yDerivative.cols();
            deriv.resize(xValue.size() * yValue.size(), derivCols);
            for (std::ptrdiff_t j = 0; j != derivCols; ++j) {
                deriv.col(j)
                    = (xValue * yDerivative.col(j).transpose()).reshaped();
            }
        } else if constexpr (!Op::hasOperandY) {
            auto const& xDerivative = Op::xPushForward();
            auto const derivCols    = xDerivative.cols();
            deriv.resize(xValue.size() * yValue.size(), derivCols);
            for (std::ptrdiff_t j = 0; j != derivCols; ++j) {
                deriv.col(j)
                    = (xDerivative.col(j) * yValue.transpose()).reshaped();
            }
        } else {
            auto const& xDerivative = Op::xPushForward();
            auto const& yDerivative = Op::yPushForward();
            auto const derivCols = xDerivative.cols(); // == yDerivative.cols()
            deriv.resize(xValue.size() * yValue.size(), derivCols);
            for (std::ptrdiff_t j = 0; j != derivCols; ++j) {
                auto const derivX
                    = (xDerivative.col(j) * yValue.transpose()).reshaped();
                auto const derivY
                    = (xValue * yDerivative.col(j).transpose()).reshaped();
                deriv.col(j) = derivX + derivY;
            }
        }

        return deriv;
    }

    template <typename OtherDerivative>
    void _pullBackImpl(OtherDerivative const& derivative)
    {
        auto const& xValue   = Op::xValue();
        auto const& yValue   = Op::yValue();
        auto const derivRows = derivative.rows();

        if constexpr (Op::hasOperandX) {
            auto deriv = Derivative(derivRows, xValue.size());
            for (std::ptrdiff_t i = 0; i != derivRows; ++i) {
                auto const matricized
                    = derivative.row(i).reshaped(xValue.size(), yValue.size());
                deriv.row(i) = (matricized * yValue).transpose();
            }
            Op::xPullBack(deriv);
        }
        if constexpr (Op::hasOperandY) {
            auto deriv = Derivative(derivRows, yValue.size());
            for (std::ptrdiff_t i = 0; i != derivRows; ++i) {
                auto const matricized
                    = derivative.row(i).reshaped(xValue.size(), yValue.size());
                deriv.row(i) = xValue.transpose() * matricized;
            }
            Op::yPullBack(deriv);
        }
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_COLVECTOR_BINARY_OP(tensorProduct, EigenAD::TensorProduct)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_TENSOR_PRODUCT_HPP
