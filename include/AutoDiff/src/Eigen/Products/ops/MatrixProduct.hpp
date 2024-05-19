// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_PRODUCT_HPP

namespace AutoDiff::EigenAD {

template <typename X, typename Y>
class MatrixProduct : public BinaryOperation<MatrixProduct<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<MatrixProduct<X, Y>, X, Y>;
    using Base::Base;
    using typename Base::Derivative;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue() * Base::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> Derivative
    {
        auto const& xValue = Base::xValue();
        auto const& yValue = Base::yValue();
        Derivative deriv;

        if constexpr (!Base::hasOperandX) {
            auto const& yDerivative = Base::yPushForward();
            auto const derivCols    = yDerivative.cols();
            deriv.resize(xValue.rows() * yValue.cols(), derivCols);
            for (std::ptrdiff_t j = 0; j != derivCols; ++j) {
                deriv.col(j) = (xValue
                                * yDerivative.col(j).reshaped(
                                    yValue.rows(), yValue.cols()))
                                   .reshaped();
            }
        } else if constexpr (!Base::hasOperandY) {
            auto const& xDerivative = Base::xPushForward();
            auto const derivCols    = xDerivative.cols();
            deriv.resize(xValue.rows() * yValue.cols(), derivCols);
            for (std::ptrdiff_t j = 0; j != derivCols; ++j) {
                deriv.col(j)
                    = (xDerivative.col(j).reshaped(xValue.rows(), xValue.cols())
                        * yValue)
                          .reshaped();
            }
        } else {
            auto const& xDerivative = Base::xPushForward();
            auto const& yDerivative = Base::yPushForward();
            auto const derivCols = xDerivative.cols(); // = yDerivative.cols()
            deriv.resize(xValue.rows() * yValue.cols(), derivCols);
            for (std::ptrdiff_t j = 0; j != derivCols; ++j) {
                auto const derivX
                    = (xDerivative.col(j).reshaped(xValue.rows(), xValue.cols())
                        * yValue)
                          .reshaped();
                auto const derivY = (xValue
                                     * yDerivative.col(j).reshaped(
                                         yValue.rows(), yValue.cols()))
                                        .reshaped();
                deriv.col(j) = derivX + derivY;
            }
        }

        return deriv;
    }

    template <typename OtherDerivative>
    void _pullBackImpl(OtherDerivative const& derivative)
    {
        auto const& xValue   = Base::xValue();
        auto const& yValue   = Base::yValue();
        auto const derivRows = derivative.rows();

        if constexpr (Base::hasOperandX) {
            auto deriv = Derivative(derivRows, xValue.size());
            for (std::ptrdiff_t i = 0; i != derivRows; ++i) {
                auto const matricized
                    = derivative.row(i).reshaped(xValue.rows(), yValue.cols());
                deriv.row(i) = (matricized * yValue.transpose()).reshaped();
            }
            Base::xPullBack(deriv);
        }
        if constexpr (Base::hasOperandY) {
            auto deriv = Derivative(derivRows, yValue.size());
            for (std::ptrdiff_t i = 0; i != derivRows; ++i) {
                auto const matricized
                    = derivative.row(i).reshaped(xValue.rows(), yValue.cols());
                deriv.row(i) = (xValue.transpose() * matricized).reshaped();
            }
            Base::yPullBack(deriv);
        }
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIX_BINARY_OP(operator*, EigenAD::MatrixProduct);

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_PRODUCT_HPP
