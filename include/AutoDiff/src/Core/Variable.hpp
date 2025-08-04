// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_CORE_VARIABLE_HPP
#define AUTODIFF_SRC_CORE_VARIABLE_HPP

#include "../internal/Node.hpp"
#include "../internal/Reference.hpp"
#include "../internal/traits.hpp"
#include "AbstractVariable.hpp"
#include "Expression.hpp"

#include <type_traits>

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
    explicit Variable(Value value);

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
    explicit Variable(Expression<Expr> const& expression);

    ~Variable() override = default;

    Variable(Variable const& other)                        = default;
    Variable(Variable&& other) noexcept                    = default;
    auto operator=(Variable const& other) -> Variable&     = default;
    auto operator=(Variable&& other) noexcept -> Variable& = default;

    /**
     * @brief Returns the cached value.
     */
    [[nodiscard]] auto operator()() const -> Value const&;

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
    template <NotVariable Expr>
    // NOLINTNEXTLINE(*-signature)
    auto operator=(Expression<Expr> const& expression) const -> Variable const&;

    /**
     * @brief Assign a literal to replace the current value or expression.
     *
     * @param value    the literal value
     */
    auto operator=(Value value) const -> Variable const&; // NOLINT(*-signature)

    /**
     * @brief Assign an expression to replace the current value or expression.

     * The new expression is immediately evaluated (eager evaluation)
     * unless macro @c AUTODIFF_NO_EAGER_EVALUATION is defined.
     *
     * @tparam Expr         the type of the expression
     * @param expression    the expression to be assigned
     */
    template <typename Expr>
    void setExpression(Expression<Expr> const& expression) const;

    /**
     * @brief Set the value of the associated derivative.
     *
     * @param  derivative  the derivative to use
     */
    void setDerivative(Derivative derivative) const;

    // Note: The following functions with leading underscores
    // are not part of the public API.

    [[nodiscard]] auto _node() const -> internal::AbstractComputation* override;

    // Expression implementation
    // ===============================================

    // Must return Value by reference to avoid dangling references to
    // temporaries in expressions!
    [[nodiscard]] auto _valueImpl() const -> Value const&;

    [[nodiscard]] auto _pushForwardImpl() const -> Derivative const&;

    template <typename OtherDerivative>
    void _pullBackImpl(OtherDerivative const& gradient) const;

    void _transferChildrenToImpl(internal::Node& node);

    void _releaseCacheImpl() const { } // does not apply to Variable

private:
    template <typename V, typename D>
    friend auto d(Variable<V, D> const&) -> D const&;

    template <typename V, typename D>
    friend auto operator==(
        Variable<V, D> const& left, Variable<V, D> const& right) -> bool;

    template <typename V, typename D>
    friend auto operator!=(
        Variable<V, D> const& left, Variable<V, D> const& right) -> bool;

    // The reference to the computation node
    // that holds the value and derivative.
    internal::Reference<Value, Derivative> mRef;
};

// free functions
// =========================================================

/**
 * @brief The differential (i.e., the cached derivative) of a variable.
 *
 * Depending on the mode of differentiation, this derivative
 * can be a tangent vector or gradient.
 *
 * @param  variable    the variable to be differentiated
 */
template <typename Value, typename Derivative>
[[nodiscard]] auto d(Variable<Value, Derivative> const& variable)
    -> Derivative const&;

/**
 * @brief Check whether two variables point to the same computation.
 */
template <typename Value, typename Derivative>
[[nodiscard]] auto operator==(Variable<Value, Derivative> const& left,
    Variable<Value, Derivative> const& right) -> bool;

/**
 * @brief Check whether two variables point to different computations.
 */
template <typename Value, typename Derivative>
[[nodiscard]] auto operator!=(Variable<Value, Derivative> const& left,
    Variable<Value, Derivative> const& right) -> bool;

// Variable factories
// =========================================================

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

} // namespace detail

/**
 * @brief Whether the value type is supported in expressions.
 */
template <typename T>
concept Evaluable = requires { typename internal::Evaluated_t<T>; };

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

template <Evaluable T>
auto var(T const& literal) -> typename detail::VariableFromValue<T>::type;

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
    typename detail::VariableFromExpr<Expr>::type;

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
    -> Variable<Value, Derivative>;

} // namespace AutoDiff

#include "Variable.tpp" // implementation

#endif // AUTODIFF_SRC_CORE_VARIABLE_HPP
