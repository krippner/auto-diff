// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_VECTOR_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_VECTOR_PRODUCT_HPP

namespace AutoDiff::EigenAD {

template <typename X, typename Y>
class MatrixVectorProduct : public Expression<MatrixVectorProduct<X, Y>>,
                            public BinaryOperation<X, Y> {
public:
    using Op = BinaryOperation<X, Y>;
    using Op::Op;
    using typename Op::Derivative;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Op::xValue() * Op::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& xValue = Op::xValue();
        auto const& yValue = Op::yValue();

        if constexpr (!Op::hasOperandX) {
            return xValue * Op::yPushForward();
        } else {
            auto const& xDerivative = Op::xPushForward();
            auto const derivCols    = xDerivative.cols();
            auto deriv              = Derivative(xValue.rows(), derivCols);
            for (std::ptrdiff_t j = 0; j != derivCols; ++j) {
                deriv.col(j)
                    = xDerivative.col(j).reshaped(xValue.rows(), xValue.cols())
                    * yValue;
            }
            if constexpr (Op::hasOperandY) {
                deriv.noalias() += xValue * Op::yPushForward();
            }
            return deriv;
        }
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
                deriv.row(i)
                    = (yValue * derivative.row(i)).transpose().reshaped();
            }
            Op::xPullBack(deriv);
        }
        if constexpr (Op::hasOperandY) {
            Op::yPullBack(derivative * xValue);
        }
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIX_COLVECTOR_OP(operator*, EigenAD::MatrixVectorProduct);

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_VECTOR_PRODUCT_HPP
