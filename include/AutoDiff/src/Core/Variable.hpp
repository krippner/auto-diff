// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_CORE_VARIABLE_HPP
#define AUTODIFF_SRC_CORE_VARIABLE_HPP

#include "../internal/Computation.hpp"
#include "../internal/Node.hpp" // NodeOwner
#include "../internal/traits.hpp"
#include "AbstractVariable.hpp"
#include "Expression.hpp"

#include <algorithm> // swap
#include <memory>
#include <type_traits> // enable_if
#include <utility>     // move

namespace AutoDiff {

/**
 * @class Variable
 * @brief Evaluates an expression and caches its value and derivative.
 *
 * The expression can be a literal or a composition of operations.
 * An AutoDiff Variable behaves similar to a mathematical variable
 * in the sense that it is essentially a label pointing to a shared resource.
 *
 * @code{.cpp}
 * auto x = var(42);      // create a literal variable with value 42
 * x();                   // get the value of x (42)
 *
 * x.set_derivative(1.0);
 * d(x);                  // get the derivative of x (1.0)
 *
 * auto y = x;            // x and y share the same resource
 * x = 3;                 // assign a new value to x
 * y();                   // y has the new value of x (3)
 * @endcode
 *
 * Variables can make computations more efficient because they allow to evaluate
 * an expression once and then reuse the cached result in other expressions.
 * Iterative computations require variables to accumulate expressions.
 *
 * @tparam Value       the type of cached value
 * @tparam Derivative  the type of cached derivative
 */
template <typename Value, typename Derivative_>
class Variable : public AbstractVariable,
                 public Expression<Variable<Value, Derivative_>> {
public:
    // used to propagate the derivative type in expressions
    using Derivative = Derivative_;

    /**
     * @brief Create a variable holding a default-constructed literal.
     */
    Variable() = default;

    // Ctors must be explicit to avoid ambiguity with assignment

    /**
     * @brief Create a variable holding a literal.
     *
     * @param value    the literal value
     */
    explicit Variable(Value value)
        : Variable{}
    {
        mRef->setValue(std::move(value));
    }

    /**
     * @brief Create a variable that evaluates an expression of other variables.
     *
     * The expression is immediately evaluated (eager evaluation)
     * unless macro @c AUTODIFF_NO_EAGER_EVALUATION is defined.
     *
     * @tparam Expr            the type of the expression
     * @param expression       the expression to be evaluated
     */
    template <typename Expr>
    explicit Variable(Expression<Expr> const& expression)
        : Variable{}
    {
        setExpression(expression);
    }

    ~Variable() override = default;

    Variable(Variable const& other)                        = default;
    Variable(Variable&& other) noexcept                    = default;
    auto operator=(Variable const& other) -> Variable&     = default;
    auto operator=(Variable&& other) noexcept -> Variable& = default;

    /**
     * @brief Returns the cached value.
     */
    [[nodiscard]] auto operator()() const -> Value const&
    {
        return mRef->value();
    }

    /**
     * @brief The differential (i.e., the cached derivative) of a variable.
     *
     * Depending on the mode of differentiation, this derivative
     * can be a tangent vector or gradient.
     *
     * @param  variable    the variable to be differentiated
     */
    [[nodiscard]] friend auto d(Variable const& variable) -> Derivative const&
    {
        return variable.mRef->derivative();
    }

    /**
     * @brief Evaluate an expression in place of the current value or
     * expression.
     *
     * The new expression is immediately evaluated (eager evaluation)
     * unless macro @c AUTODIFF_NO_EAGER_EVALUATION is defined.
     *
     * @warning The expression must not contain this variable.
     * If it does, the @c Function it is added to
     * will throw a @c CyclicDependencyError.
     *
     * @tparam Expr        the type of the expression, must not be Variable
     * @param expression   the expression to be evaluated
     */
    template <typename Expr,
        typename = std::enable_if_t<!std::is_same_v<Expr, Variable>>>
    auto operator=( // NOLINT(*-signature)
        Expression<Expr> const& expression) const -> Variable const&
    {
        setExpression(expression);
        return *this;
    }

    /**
     * @brief Assign a literal to replace the current value or expression.
     *
     * @param value    the literal value
     */
    auto operator=(Value value) const -> Variable const& // NOLINT(*-signature)
    {
        mRef->setValue(std::move(value));
        return *this;
    }

    /**
     * @brief Assign an expression to replace the current value or expression.

     * The new expression is immediately evaluated (eager evaluation)
     * unless macro @c AUTODIFF_NO_EAGER_EVALUATION is defined.
     *
     * @tparam Expr         the type of the expression
     * @param expression    the expression to be assigned
     */
    template <typename Expr>
    void setExpression(Expression<Expr> const& expression) const
    {
        mRef->setExpression(expression);
#ifndef AUTODIFF_NO_EAGER_EVALUATION
        mRef->evaluate();
#endif
    }

    /**
     * @brief Check whether two variables point to the same computation.
     */
    [[nodiscard]] friend auto operator==(
        Variable const& left, Variable const& right) -> bool
    {
        return left.mRef == right.mRef;
    }

    /**
     * @brief Check whether two variables point to different computations.
     */
    [[nodiscard]] friend auto operator!=(
        Variable const& left, Variable const& right) -> bool
    {
        return left.mRef != right.mRef;
    }

    /**
     * @brief Set the value of the associated derivative.
     *
     * @param  derivative  the derivative to use
     */
    void setDerivative(Derivative derivative) const
    {
        mRef->setDerivative(std::move(derivative));
    }

    // Note: The following functions with leading underscores
    // are not part of the public API.

    [[nodiscard]] auto _node() const -> internal::AbstractComputation* override
    {
        return mRef.operator->();
    }

    // Expression implementation ===============================================

    // Must return Value by reference to avoid dangling references to
    // temporaries in expressions!
    [[nodiscard]] auto _valueImpl() const -> Value const&
    {
        return mRef->value();
    }

    [[nodiscard]] auto _pushForwardImpl() const -> Derivative const&
    {
        return mRef->derivative();
    }

    template <typename OtherDerivative>
    void _pullBackImpl(OtherDerivative const& gradient) const
    {
        mRef->addGradient(gradient);
    }

    void _transferChildrenToImpl(internal::Node& node)
    {
        mRef.transferOperationTo(node);
    }

    void _releaseCacheImpl() const { } // does not apply to Variable

private:
    /**
     * @class Reference
     * @brief Essentially a shared pointer to a computation node.
     *
     * Additionally, it holds a unique owner object that is used
     * to register and unregister ownership of the computation.
     */
    class Reference {
    public:
        using Owner       = internal::NodeOwner;
        using Computation = internal::Computation<Value, Derivative>;

        Reference() { mComputation->addParentOwner(mOwner); }

        Reference(Reference const& other)
            : mComputation{other.mComputation}
            , mComputationPtr{other.mComputationPtr}
        {
            mComputation->addParentOwner(mOwner); // owner is unique
        }

        auto operator=(Reference other) -> Reference&
        {
            swap(*this, other);
            return *this;
        }

        ~Reference()
        {
            if (static_cast<bool>(mComputation)) {
                mComputation->removeParentOwner(mOwner);
            } // else transferOperationTo was called
        }

        Reference(Reference&&) noexcept                    = default;
        auto operator=(Reference&&) noexcept -> Reference& = default;

        [[nodiscard]] auto operator->() const -> Computation*
        {
            // Note: raw ptr always valid:
            // ~Node guarantees that this function is not called
            // between ~Computation and ~Variable.
            return mComputationPtr;
        }

        void transferOperationTo(internal::Node& node)
        {
            // Note: shared_ptr always valid:
            // This function is called at most once, which is when the
            // parent Variable is bound in an expression (copy ctor).

            // unregister ownership
            mComputation->removeParentOwner(mOwner);
            // transfer owning pointer to node
            node.addChild(mComputation);
            mComputation.reset();
        }

        [[nodiscard]] friend auto operator==(
            Reference const& left, Reference const& right)
        {
            return left.mComputationPtr == right.mComputationPtr;
        }

        [[nodiscard]] friend auto operator!=(
            Reference const& left, Reference const& right)
        {
            return left.mComputationPtr != right.mComputationPtr;
        }

        friend void swap(Reference& a, Reference& b) noexcept
        {
            using std::swap;
            swap(a.mOwner, b.mOwner);
            swap(a.mComputation, b.mComputation);
            swap(a.mComputationPtr, b.mComputationPtr);
        }

    private:
        std::unique_ptr<Owner> mOwner{std::make_unique<Owner>()};
        std::shared_ptr<Computation> mComputation{
            std::make_shared<Computation>()};
        Computation* mComputationPtr{mComputation.get()};
    };

    Reference mRef;
};

// Variable factories =========================================================

namespace detail {

    // The following traits are used to determine the value and derivative.

    template <typename T>
    struct VariableFromValue {
        using Value      = internal::Evaluated_t<T>;
        using Derivative = internal::DefaultDerivative_t<Value>;
        using type       = Variable<Value, Derivative>;
    };

    template <typename Expr>
    struct VariableFromExpr {
        using Value      = internal::Evaluated_t<ValueType_t<Expr>>;
        using Derivative = typename Expr::Derivative;
        using type       = Variable<Value, Derivative>;
    };

    template <typename T, typename = std::void_t<>>
    struct IsSupportedValue : std::false_type { };

    template <typename T>
    struct IsSupportedValue<T, std::void_t<internal::Evaluated_t<T>>>
        : std::true_type { };

} // namespace detail

/**
 * @brief Whether the value type is supported in expressions.
 */
template <typename T>
constexpr bool isSupportedValue_v = detail::IsSupportedValue<T>::value;

/**
 * @brief Create a variable holding a literal.
 *
 * The literal is evaluated before it is assigned.
 *
 * @note The value and derivative type of the resulting variable
 * depend on the implementation supplied by a module.
 *
 * @tparam T           the type of the unevaluated literal
 * @param  literal     the literal value
 */
template <typename T, typename = std::enable_if_t<isSupportedValue_v<T>>>
auto var(T const& literal) -> typename detail::VariableFromValue<T>::type
{
    return typename detail::VariableFromValue<T>::type(literal);
}

/**
 * @brief Create a variable that evaluates an expression of other variables.
 *
 * The expression is immediately evaluated (eager evaluation)
 * unless macro @c AUTODIFF_NO_EAGER_EVALUATION is defined.
 *
 * @note The value and derivative type of the resulting variable
 * depend on the implementation supplied by a module.
 *
 * @tparam Expr            the type of the expression
 * @param expression       the expression to be evaluated
 */
template <typename Expr>
auto var(Expression<Expr> const& expression) ->
    typename detail::VariableFromExpr<Expr>::type
{
    return typename detail::VariableFromExpr<Expr>::type(expression);
}

/**
 * @brief Create a variable that depends on another variable
 * through the identity.
 *
 * @tparam Value           the value type of the variable
 * @tparam Derivative      the derivative type of the variable
 * @param  variable        the variable to depend on
 */
template <typename Value, typename Derivative>
auto var(Variable<Value, Derivative> const& variable)
{
    Variable<Value, Derivative> newVariable;
    newVariable.setExpression(variable);
    return newVariable;
}

} // namespace AutoDiff

#endif // AUTODIFF_SRC_CORE_VARIABLE_HPP
