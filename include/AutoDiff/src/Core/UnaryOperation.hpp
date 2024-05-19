// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_CORE_UNARY_OPERATION_HPP
#define AUTODIFF_SRC_CORE_UNARY_OPERATION_HPP

#include "Expression.hpp"

namespace AutoDiff {

/**
 * @class UnaryOperation
 * @brief Auxiliary base class for operations depending on one other expression.
 *
 * A subclass can simply reuse the constructor "using Base::Base".
 * The derived class must implement the public functions
 * @c _valueImpl       (returning its value),
 * @c _pushForwardImpl (returning the pushforward of the tangent vector), and
 * @c _pullBackImpl    (pushing back the gradient).
 * For details, see the @c Expression class.
 *
 * @tparam Derived  the derived class of the expression, e.g. Basic::Exp<X>
 * @tparam X        the derived class of the operand
 */
template <typename Derived, typename X>
class UnaryOperation : public Expression<Derived> {
public:
    using Derivative = typename X::Derivative; // propagate the derivative type

    /**
     * @brief Create a unary operation that stores a copy of the operand.
     *
     * @param  operand     the operand
     */
    explicit UnaryOperation(Expression<X> const& operand)
        : mOperand{operand.derived()}
    {
    }

    // Expression implementation ===============================================

    void _transferChildrenToImpl(internal::Node& node)
    {
        mOperand._transferChildrenTo(node);
    }

    void _releaseCacheImpl() { mOperand._releaseCache(); }

protected:
    ~UnaryOperation() = default;

    UnaryOperation(UnaryOperation const&)                        = default;
    UnaryOperation(UnaryOperation&&) noexcept                    = default;
    auto operator=(UnaryOperation const&) -> UnaryOperation&     = default;
    auto operator=(UnaryOperation&&) noexcept -> UnaryOperation& = default;

    /**
     * @brief Compute the value of the operand.
     */
    auto xValue() -> decltype(auto) { return mOperand._value(); }

    /**
     * @brief Compute the pushforward by the operand.
     */
    auto xPushForward() -> decltype(auto) { return mOperand._pushForward(); }

    /**
     * @brief Pull back the gradient by the operand.
     */
    template <typename Derivative>
    void xPullBack(Derivative const& derivative)
    {
        mOperand._pullBack(derivative);
    }

private:
    X mOperand;
};

} // namespace AutoDiff

#endif // AUTODIFF_SRC_CORE_UNARY_OPERATION_HPP
