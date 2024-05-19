// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_INTERNAL_COMPUTATION_HPP
#define AUTODIFF_SRC_INTERNAL_COMPUTATION_HPP

#include "AbstractComputation.hpp"
#include "Evaluator.hpp"
#include "TypeImpl.hpp"

#include <cassert>
#include <memory>
#include <type_traits> // is_same
#include <utility>     // move

namespace AutoDiff::internal {

/**
 * @class Computation
 * @brief Implementation of a computation node.
 *
 * Evaluates matching expression templates to a specific value type
 * and derivative type and caches the results.
 * Expression can be literals or nested expressions.
 *
 * @tparam Value       the value type
 * @tparam Derivative  the derivative type
 */
template <typename Value, typename Derivative>
class Computation : public AbstractComputation {
public:
    /**
     * @brief Make this a literal computation.
     *
     * Stores the new value and releases ownership of child computations
     * bound in the previous expression.
     *
     * @param  value       the value of the literal
     */
    void setValue(Value value)
    {
        releaseChildren();
        mEvaluator.release(); // NOLINT(*-unused-return-value)
        mValue = std::move(value);
    }

    /**
     * @brief Make this an expression computation.
     *
     * Stores the new expression and takes ownership of its child computations.
     * Releases ownership of child computations bound in the previous
     * expression.
     *
     * @note The expression must have matching value and derivative types.
     *
     * @tparam Expr          the derived type of the expression
     * @param  expression    the expression to be evaluated
     */
    template <typename Expr>
    void setExpression(Expression<Expr> const& expression)
    {
        static_assert(std::is_same_v<Value, typename Evaluator<Expr>::Value>,
            "EXPRESSION VALUE TYPE MISMATCH");
        static_assert(
            std::is_same_v<Derivative, typename Evaluator<Expr>::Derivative>,
            "EXPRESSION DERIVATIVE TYPE MISMATCH");

        releaseChildren();
        mEvaluator = std::make_unique<Evaluator<Expr>>(expression);
        mEvaluator->transferChildrenTo(*this);
    }

    /**
     * @brief The cached value.
     */
    [[nodiscard]] auto value() const -> Value const& { return mValue; }

    /**
     * @brief The cached derivative.
     *
     * Evaluates the derivative if necessary.
     */
    auto derivative() -> Derivative const&
    {
        if (mDerivativeDescr.state != MapDescription::evaluated) {
            // lazy evaluation
            generate(mDerivative, mDerivativeDescr);
            mDerivativeDescr.state = MapDescription::evaluated;
        }
        return mDerivative;
    }

    /**
     * @brief Manually set the derivative.
     *
     * @param  derivative  the derivative to store
     */
    void setDerivative(Derivative derivative)
    {
        mDerivative            = std::move(derivative);
        mDerivativeDescr.state = MapDescription::evaluated;
    }

    /**
     * @brief Add a gradient to the current derivative.
     *
     * The @c Derivative type must support assignment and add-assignment
     * from @c OtherDerivative type. See TypeImpl.hpp for details.
     *
     * @param  gradient    the gradient to be added
     */
    template <typename OtherDerivative>
    void addGradient(OtherDerivative const& gradient)
    {
        if (mDerivativeDescr.state == MapDescription::zero) {
            assign(mDerivative, gradient);
            mDerivativeDescr.state = MapDescription::evaluated;
        } else {
            addTo(mDerivative, gradient);
        }
    }

    // AbstractComputation implementation ====================================

    void evaluate() final
    {
        assert(mEvaluator && "LITERALS CANNOT BE EVALUATED");
        mEvaluator->evaluateTo(mValue);
    }

#ifndef AUTODIFF_NO_FORWARD_MODE
    void setTangentZero(Shape domainShape) final
    {
        mDerivativeDescr.state         = MapDescription::zero;
        mDerivativeDescr.domainShape   = domainShape;
        mDerivativeDescr.codomainShape = getShape(mValue);
    }

    void pushTangent() final
    {
        assert(mEvaluator && "LITERALS CANNOT BE DIFFERENTIATED");
        mEvaluator->pushForwardTo(mDerivative);
        mDerivativeDescr.state = MapDescription::evaluated;
    }
#endif

#ifndef AUTODIFF_NO_REVERSE_MODE
    void setGradientZero(Shape codomainShape) final
    {
        mDerivativeDescr.state         = MapDescription::zero;
        mDerivativeDescr.domainShape   = getShape(mValue);
        mDerivativeDescr.codomainShape = codomainShape;
    }

    void pullGradient() final
    {
        assert(mEvaluator && "LITERALS CANNOT BE DIFFERENTIATED");
        mEvaluator->pullBack(derivative());
    }
#endif

    void setDerivativeIdentity() final
    {
        mDerivativeDescr.state         = MapDescription::identity;
        auto const shape               = getShape(mValue);
        mDerivativeDescr.domainShape   = shape;
        mDerivativeDescr.codomainShape = shape;
    }

    [[nodiscard]] auto valueShape() const -> Shape final
    {
        return getShape(mValue);
    }

    [[nodiscard]] auto derivativeCodomainShape() const -> Shape final
    {
        return codomainShape(mDerivative);
    }

private:
    Value mValue{};
    Derivative mDerivative{};
    MapDescription mDerivativeDescr{};

    std::unique_ptr<AbstractEvaluator<Value, Derivative>> mEvaluator{};
};

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_COMPUTATION_HPP
