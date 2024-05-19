// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SCR_CORE_EXPRESSION_HPP
#define AUTODIFF_SCR_CORE_EXPRESSION_HPP

#include <type_traits>
#include <utility> // declval

namespace AutoDiff {

namespace internal {

    class Node;

} // namespace internal

/**
 * @class Expression
 * @brief Base class for expressions in AutoDiff.
 *
 * This class defines the common interface that derived expressions
 * must implement. Expressions in AutoDiff are variables or operations.
 * This class uses CRTP (Curiously Recurring Template Pattern) to achieve
 * compile-time polymorphism. It is trivial and used for overload resolution.
 *
 * @note The member functions with leading underscore are internal and
 * should not be called by the user. They are used by other operations to
 * evaluate the expression, compute the pushforward, or pull back gradients.
 *
 * @tparam Derived      The derived expression type, e.g. Basic::Sum<X,Y>
 */
template <typename Derived>
class Expression {
public:
    ~Expression() = default;

    Expression(Expression const&)     = default;
    Expression(Expression&&) noexcept = default;

    auto operator=(Expression const&) -> Expression&     = default;
    auto operator=(Expression&&) noexcept -> Expression& = default;

    /**
     * @brief Get a reference to the derived object.
     */
    [[nodiscard]] auto derived() const -> Derived const&
    {
        // cast is safe, static_assert in constructor
        return static_cast<Derived const&>(*this);
    }

    /**
     * @brief Get a reference to the derived object.
     */
    [[nodiscard]] auto derived() -> Derived&
    {
        // cast is safe, static_assert in constructor
        return static_cast<Derived&>(*this);
    }

    /**
     * @brief Compute the value of this expression.
     */
    [[nodiscard]] auto _value() -> decltype(auto)
    {
        return derived()._valueImpl();
    }

#ifndef AUTODIFF_NO_FORWARD_MODE
    /**
     * @brief Compute the pushforward of the current tangent by this expression.
     *
     * The pushforward is a linear map from tangent vectors on the domain of
     * this operation to tangent vectors on its codomain.
     */
    [[nodiscard]] auto _pushForward() -> decltype(auto)
    {
        return derived()._pushForwardImpl();
    }
#endif

#ifndef AUTODIFF_NO_REVERSE_MODE
    /**
     * @brief Compute the pullback of the current gradient by this expression.
     *
     * The pullback is a linear map from gradients on the codomain of this
     * operation to gradients on its domain.
     *
     * @tparam Derivative   The type of the gradient/derivative
     * @param gradient      The gradient to pull back
     */
    template <typename Derivative>
    void _pullBack(Derivative const& gradient)
    {
        derived()._pullBackImpl(gradient);
    }
#endif

    /**
     * @brief Transfer ownership of the child nodes to the parent node.
     *
     * The parent node, see class @c internal::Computation, owns and evaluates
     * a composition of operations.
     * This method recursively adds the input nodes of this composition
     * to the parent node, building a computation graph. Transferring ownership
     * is necessary for the iterative destruction of the graph.
     *
     * @param node      The parent node
     */
    void _transferChildrenTo(internal::Node& node)
    {
        derived()._transferChildrenToImpl(node);
    }

    /**
     * @brief Release temporary data that has been cached during evaluation.
     *
     * Needed, e.g., for the Python bindings, where every operation
     * must temporarily cache its result.
     */
    void _releaseCache() { derived()._releaseCacheImpl(); }

protected:
    // only derived classes can be instantiated
    Expression()
    {
        static_assert(std::is_base_of_v<Expression<Derived>, Derived>,
            "Derived MUST DERIVE FROM Expression<Derived>");
    }
};

/**
 * @brief Type trait to check if a type is an Expression.
 */
template <typename Expr>
constexpr bool isExpression_v = std::is_base_of_v<Expression<Expr>, Expr>;

namespace detail {

    /**
     * @brief Alias template to remove cv-qualifiers and references from a type.
     */
    template <typename T>
    using Unqualified_t = std::remove_cv_t<std::remove_reference_t<T>>;

    /**
     * @brief Type trait to get the value type of an Expression.
     */
    template <typename T, typename = void>
    struct ValueType {
        using type = Unqualified_t<T>; // default when T is not an Expression
    };

    template <typename T>
    struct ValueType<T, std::void_t<decltype(std::declval<T>()._value())>> {
        using type = Unqualified_t<decltype(std::declval<T>()._value())>;
    };

} // namespace detail

/**
 * @brief Alias template to get the value type of an Expression.
 */
template <typename Expr>
using ValueType_t = typename detail::ValueType<Expr>::type;

} // namespace AutoDiff

#endif // AUTODIFF_SRC_CORE_EXPRESSION_HPP
