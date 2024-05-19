// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_VECTOR_PRODUCT_HPP
#define AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_VECTOR_PRODUCT_HPP

namespace AutoDiff::EigenAD {

template <typename X, typename Y>
class MatrixVectorProduct
    : public BinaryOperation<MatrixVectorProduct<X, Y>, X, Y> {
public:
    using Base = BinaryOperation<MatrixVectorProduct<X, Y>, X, Y>;
    using Base::Base;
    using typename Base::Derivative;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue() * Base::yValue();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        auto const& xValue = Base::xValue();
        auto const& yValue = Base::yValue();

        if constexpr (!Base::hasOperandX) {
            return xValue * Base::yPushForward();
        } else {
            auto const& xDerivative = Base::xPushForward();
            auto const derivCols    = xDerivative.cols();
            auto deriv              = Derivative(xValue.rows(), derivCols);
            for (std::ptrdiff_t j = 0; j != derivCols; ++j) {
                deriv.col(j)
                    = xDerivative.col(j).reshaped(xValue.rows(), xValue.cols())
                    * yValue;
            }
            if constexpr (Base::hasOperandY) {
                deriv.noalias() += xValue * Base::yPushForward();
            }
            return deriv;
        }
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
                deriv.row(i)
                    = (yValue * derivative.row(i)).transpose().reshaped();
            }
            Base::xPullBack(deriv);
        }
        if constexpr (Base::hasOperandY) {
            Base::yPullBack(derivative * xValue);
        }
    }
};

} // namespace AutoDiff::EigenAD

namespace AutoDiff {

AUTODIFF_MAKE_MATRIX_COLVECTOR_OP(operator*, EigenAD::MatrixVectorProduct);

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_PRODUCTS_OPS_MATRIX_VECTOR_PRODUCT_HPP
