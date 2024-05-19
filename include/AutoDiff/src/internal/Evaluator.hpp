// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_INTERNAL_EVALUATOR_HPP
#define AUTODIFF_SRC_INTERNAL_EVALUATOR_HPP

#include "../Core/Expression.hpp"
#include "TypeImpl.hpp"
#include "traits.hpp"

#include <utility> // move

namespace AutoDiff::internal {

class Node;

/**
 * @class AbstractEvaluator
 * @brief Abstract base class for evaluators with a specific value type and
 * derivative type.
 *
 * Derived evaluators are responsible for storing an <em>expression
 * template</em> and for evaluating it to a matching value and derivative type.
 *
 * @tparam Value_          the value type
 * @tparam Derivative_     the derivative type
 */
template <typename Value_, typename Derivative_>
class AbstractEvaluator {
public:
    using Value      = Value_;
    using Derivative = Derivative_;

    virtual ~AbstractEvaluator() = default;

    /**
     * @brief Transfer the ownership of the child (computation) nodes
     * of the contained expression to a parent node.
     *
     * @param  node        the parent node
     */
    virtual void transferChildrenTo(Node& node) = 0;

    /**
     * @brief Evaluate the value of the expression.
     *
     * @param  value       the value is assigned to this object
     */
    virtual void evaluateTo(Value& value) = 0;

#ifndef AUTODIFF_NO_FORWARD_MODE
    /**
     * @brief Push forward the tangents by the expression.
     *
     * @param  tangent  the accumulated derivative is assigned to this object
     */
    virtual void pushForwardTo(Derivative& tangent) = 0;
#endif

#ifndef AUTODIFF_NO_REVERSE_MODE
    /**
     * @brief Pull back a gradient by the expression.
     *
     * @param  gradient    the gradient to be pulled back
     */
    virtual void pullBack(Derivative const& gradient) = 0;
#endif

protected:
    AbstractEvaluator()                                            = default;
    AbstractEvaluator(AbstractEvaluator const&)                    = default;
    AbstractEvaluator(AbstractEvaluator&&) noexcept                = default;
    auto operator=(AbstractEvaluator const&) -> AbstractEvaluator& = default;
    auto operator=(
        AbstractEvaluator&&) noexcept -> AbstractEvaluator& = default;
};

template <typename Expr>
struct EvaluatorType {
    using Value      = Evaluated_t<ValueType_t<Expr>>;
    using Derivative = typename Expr::Derivative;
    using type       = AbstractEvaluator<Value, Derivative>;
};

/**
 * @brief The base type of an evaluator for an expression.
 *
 * This type trait derives the value and derivative type an expression
 * can be evaluated to.
 *
 * @tparam Expr    the derived type of the expression to be evaluated
 */
template <typename Expr>
using EvaluatorType_t = typename EvaluatorType<Expr>::type;

/**
 * @class Evaluator
 * @brief Stores an expression template and evaluates its value and derivative.
 *
 * Being templated with the derived type of an expression, this class
 * allows to optimize expression evaluation at compile-time.
 *
 * @tparam Expr    the derived type of expression
 */
template <typename Expr>
class Evaluator : public EvaluatorType_t<Expr> {
public:
    using Base = EvaluatorType_t<Expr>;
    using typename Base::Derivative;
    using typename Base::Value;

    /**
     * @brief Create an evaluator for an expression.
     *
     * Stores a copy of the expression template.
     *
     * @param expression     the expression to be evaluated
     */
    explicit Evaluator(Expression<Expr> const& expression)
        : mExpression{expression.derived()}
    {
    }

    ~Evaluator() override = default;

    Evaluator(Evaluator const&)                        = default;
    Evaluator(Evaluator&&) noexcept                    = default;
    auto operator=(Evaluator const&) -> Evaluator&     = default;
    auto operator=(Evaluator&&) noexcept -> Evaluator& = default;

    void transferChildrenTo(Node& node) final
    {
        mExpression._transferChildrenTo(node);
    }

    void evaluateTo(Value& value) final
    {
        assign(value, mExpression._value());
        mExpression._releaseCache();
    }

#ifndef AUTODIFF_NO_FORWARD_MODE
    void pushForwardTo(Derivative& tangent) final
    {
        assign(tangent, mExpression._pushForward());
        mExpression._releaseCache();
    }
#endif

#ifndef AUTODIFF_NO_REVERSE_MODE
    void pullBack(Derivative const& gradient) final
    {
        mExpression._pullBack(gradient);
    }
#endif

private:
    Expr mExpression;
};

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_EVALUATOR_HPP
