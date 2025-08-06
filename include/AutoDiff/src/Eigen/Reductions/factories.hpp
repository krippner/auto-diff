// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file factories.hpp
 * @brief Macro defining a factory function for reductions of Eigen matrices.
 *
 * Use these macros inside the @c AutoDiff namespace. They can be used without
 * namespace qualification in user code because of argument-dependent lookup.
 */

#ifndef AUTODIFF_SRC_EIGEN_REDUCTIONS_FACTORIES_HPP
#define AUTODIFF_SRC_EIGEN_REDUCTIONS_FACTORIES_HPP

#include "../concepts.hpp"

#define AUTODIFF_MAKE_MATRIXBASE_UNARY_OP(operation, Type)                     \
    template <EigenAD::MatrixBaseExpression X>                                 \
    auto operation(Expression<X> const& x)                                     \
    {                                                                          \
        return Type<X>(x);                                                     \
    }

#endif // AUTODIFF_SRC_EIGEN_REDUCTIONS_FACTORIES_HPP
