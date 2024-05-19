// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_CORE_BINARY_OPERATION_HPP
#define AUTODIFF_SRC_CORE_BINARY_OPERATION_HPP

#include "Expression.hpp"
#include "UnaryOperation.hpp"

#include <type_traits> // enable_if
#include <utility>     // move

namespace AutoDiff {

/**
 * @class BinaryOperation
 * @brief Auxiliary base class for operations with two operands.
 *
 * One of the operands may be a literal instead of an expression.
 *
 * A subclass might reuse the constructor "using Base::Base".
 * The derived class must implement the public functions
 * @c _valueImpl       (returning its value),
 * @c _pushForwardImpl (returning the pushforward of the tangent vector), and
 * @c _pullBackImpl    (pushing back the gradient).
 * For details, see the @c Expression class.
 *
 * @tparam Derived  the derived class of the expression, e.g. Basic::Sum<X, Y>
 * @tparam X        the type of literal or expression of the first operand
 * @tparam Y        the type of literal or expression of the second operand
 */
template <typename Derived, typename X, typename Y, typename = void>
class BinaryOperation;

/**
 * @brief Specialization for operands (Expression, Expression).
 */
template <typename Derived, typename X, typename Y>
class BinaryOperation<Derived, X, Y,
    std::enable_if_t<isExpression_v<X> && isExpression_v<Y>>>
    : public Expression<Derived> {
public:
    using Derivative = typename X::Derivative; // propagate the derivative type

    /**
     * @brief Create a binary operation that stores copies of the operands.
     *
     * @note The operands must have the same derivative type.
     *
     * @param  operandX    the first operand
     * @param  operandY    the second operand
     */
    BinaryOperation(
        Expression<X> const& operandX, Expression<Y> const& operandY)
        : mOperandX{operandX.derived()}
        , mOperandY{operandY.derived()}
    {
        static_assert(std::is_same_v<Derivative, typename Y::Derivative>,
            "OPERANDS MUST HAVE THE SAME DERIVATIVE TYPE");
    }

    // Expression implementation ===============================================

    void _transferChildrenToImpl(internal::Node& node)
    {
        mOperandX._transferChildrenTo(node);
        mOperandY._transferChildrenTo(node);
    }

    void _releaseCacheImpl()
    {
        mOperandX._releaseCache();
        mOperandY._releaseCache();
    }

protected:
    // Wether the first operand is an expression rather than a literal
    static constexpr bool hasOperandX = true;
    // Wether the second operand is an expression rather than a literal
    static constexpr bool hasOperandY = true;

    ~BinaryOperation() = default;

    BinaryOperation(BinaryOperation const&)                        = default;
    BinaryOperation(BinaryOperation&&) noexcept                    = default;
    auto operator=(BinaryOperation const&) -> BinaryOperation&     = default;
    auto operator=(BinaryOperation&&) noexcept -> BinaryOperation& = default;

    /**
     * @brief Compute the value of the first operand.
     */
    auto xValue() -> decltype(auto) { return mOperandX._value(); }
    /**
     * @brief Compute the value of the second operand.
     */
    auto yValue() -> decltype(auto) { return mOperandY._value(); }

    /**
     * @brief Compute the pushforward by the first operand.
     */
    auto xPushForward() -> decltype(auto) { return mOperandX._pushForward(); }
    /**
     * @brief Compute the pushforward by the second operand.
     */
    auto yPushForward() -> decltype(auto) { return mOperandY._pushForward(); }

    /**
     * @brief Pull back the gradient by the first operand.
     */
    template <typename Derivative>
    void xPullBack(Derivative const& derivative)
    {
        mOperandX._pullBack(derivative);
    }

    /**
     * @brief Pull back the gradient by the second operand.
     */
    template <typename Derivative>
    void yPullBack(Derivative const& derivative)
    {
        mOperandY._pullBack(derivative);
    }

private:
    X mOperandX;
    Y mOperandY;
};

/**
 * @brief Specialization for operands (literal, Expression).
 */
template <typename Derived, typename XValue, typename Y>
class BinaryOperation<Derived, XValue, Y,
    std::enable_if_t<!isExpression_v<XValue> && isExpression_v<Y>>>
    : public UnaryOperation<Derived, Y> {
public:
    using Base = UnaryOperation<Derived, Y>;
    using typename Base::Derivative;

    BinaryOperation(XValue xValue, Expression<Y> const& operandY)
        : Base{operandY}
        , mXValue{std::move(xValue)}
    {
    }

protected:
    static constexpr bool hasOperandX = false;
    static constexpr bool hasOperandY = true;

    ~BinaryOperation() = default;

    BinaryOperation(BinaryOperation const&)                        = default;
    BinaryOperation(BinaryOperation&&) noexcept                    = default;
    auto operator=(BinaryOperation const&) -> BinaryOperation&     = default;
    auto operator=(BinaryOperation&&) noexcept -> BinaryOperation& = default;

    [[nodiscard]] auto xValue() const -> XValue const& { return mXValue; }
    auto yValue() -> decltype(auto) { return Base::xValue(); }

    auto yPushForward() -> decltype(auto) { return Base::xPushForward(); }

    template <typename Derivative>
    void yPullBack(Derivative const& derivative)
    {
        Base::xPullBack(derivative);
    }

private:
    // Value is moved into class and move constructed when this operation
    // is passed as rvalue to another operation
    XValue mXValue;
};

/**
 * @brief Specialization for operands (Expression, literal).
 */
template <typename Derived, typename X, typename YValue>
class BinaryOperation<Derived, X, YValue,
    std::enable_if_t<isExpression_v<X> && !isExpression_v<YValue>>>
    : public UnaryOperation<Derived, X> {
public:
    using Base = UnaryOperation<Derived, X>;
    using typename Base::Derivative;

    BinaryOperation(Expression<X> const& operandX, YValue yValue)
        : Base{operandX}
        , mYValue{std::move(yValue)}
    {
    }

protected:
    static constexpr bool hasOperandX = true;
    static constexpr bool hasOperandY = false;

    ~BinaryOperation() = default;

    BinaryOperation(BinaryOperation const&)                        = default;
    BinaryOperation(BinaryOperation&&) noexcept                    = default;
    auto operator=(BinaryOperation const&) -> BinaryOperation&     = default;
    auto operator=(BinaryOperation&&) noexcept -> BinaryOperation& = default;

    auto xValue() -> decltype(auto) { return Base::xValue(); }
    [[nodiscard]] auto yValue() const -> YValue const& { return mYValue; }

    auto xPushForward() -> decltype(auto) { return Base::xPushForward(); }

    template <typename Derivative>
    void xPullBack(Derivative const& derivative)
    {
        Base::xPullBack(derivative);
    }

private:
    YValue mYValue;
};

} // namespace AutoDiff

#endif // AUTODIFF_SRC_CORE_BINARY_OPERATION_HPP
