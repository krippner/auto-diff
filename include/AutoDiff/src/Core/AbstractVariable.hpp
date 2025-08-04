// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_ABSTRACT_VARIABLE_HPP
#define AUTODIFF_ABSTRACT_VARIABLE_HPP

#include "../internal/AbstractComputation.hpp"

namespace AutoDiff {

/**
 * @class AbstractVariable
 * @brief Abstract base class for all variables.
 */
class AbstractVariable {
public:
    /**
     * @brief Pointer to the internal computation node.
     *
     * The @c internal::AbstractComputation node provides access
     * to the internal computation graph.
     * It is used by the @c Function class to evaluate programs
     * and compute their derivatives.
     */
    [[nodiscard]] virtual auto _node() const
        -> internal::AbstractComputation* = 0;

    virtual ~AbstractVariable() = default;

protected:
    // only derived classes can be instantiated
    AbstractVariable() = default;

    AbstractVariable(AbstractVariable const&)                        = default;
    AbstractVariable(AbstractVariable&&) noexcept                    = default;
    auto operator=(AbstractVariable const&) -> AbstractVariable&     = default;
    auto operator=(AbstractVariable&&) noexcept -> AbstractVariable& = default;
};

} // namespace AutoDiff

#endif // AUTODIFF_ABSTRACT_VARIABLE_HPP
