// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/**
 * @file ops.hpp
 * @brief Includes supported operations for basic (arithmetic) types.
 */

#ifndef AUTODIFF_SRC_BASIC_OPS_HPP
#define AUTODIFF_SRC_BASIC_OPS_HPP

// avoid includes in operation headers
#include "common.hpp"

#include "ops/Cos.hpp"
#include "ops/Difference.hpp"
#include "ops/Exp.hpp"
#include "ops/Log.hpp"
#include "ops/Max.hpp"
#include "ops/Min.hpp"
#include "ops/Negation.hpp"
#include "ops/Pow.hpp"
#include "ops/Product.hpp"
#include "ops/Quotient.hpp"
#include "ops/Sin.hpp"
#include "ops/Sqrt.hpp"
#include "ops/Square.hpp"
#include "ops/Sum.hpp"

#endif // AUTODIFF_SRC_BASIC_OPS_HPP
