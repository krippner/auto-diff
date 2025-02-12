// Copyright (c) 2024-2025 Matthias Krippner
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
 * This class augments the derived class with the operand and provides
 * a constructor that the derived class can reuse e.g. "using Op::Op".
 * Here is a typical example of a derived class:
 * @code{.cpp}
 * template <typename X>
 * class Negation : public Expression<Negation<X>>, public UnaryOperation<X> {
 * public:
 *   using Op = UnaryOperation<X>;
 *   using Op::Op; // reuse the constructor
 *
 *   auto _valueImpl() -> decltype(auto) {
 *     return -Op::xValue(); // access the operand value
 *   }
 *
 *   auto _pushForwardImpl() -> decltype(auto) {
 *     return -Op::xPushForward(); // access the pushforward by the operand
 *   }
 *
 *   template <typename Derivative>
 *   void _pullBackImpl(Derivative const& derivative) {
 *     Op::xPullBack(-derivative); // pull back the gradient
 *   }
 * };
 * @endcode
 *
 * @tparam X        the derived class of the operand
 */
template <typename X>
class UnaryOperation {
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
