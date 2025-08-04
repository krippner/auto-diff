#include <utility> // move

namespace AutoDiff {

template <typename Value, typename Derivative>
Variable<Value, Derivative>::Variable(Value value)
    : Variable{}
{
    mRef->setValue(std::move(value));
}

template <typename Value, typename Derivative>
template <typename Expr>
Variable<Value, Derivative>::Variable(Expression<Expr> const& expression)
    : Variable{}
{
    setExpression(expression);
}

template <typename Value, typename Derivative>
auto Variable<Value, Derivative>::operator()() const -> Value const&
{
    return mRef->value();
}

template <typename Value, typename Derivative>
template <NotVariable Expr>
auto Variable<Value, Derivative>::operator=(
    Expression<Expr> const& expression) const -> Variable const&
{
    setExpression(expression);
    return *this;
}

template <typename Value, typename Derivative>
auto Variable<Value, Derivative>::operator=(Value value) const
    -> Variable const&
{
    mRef->setValue(std::move(value));
    return *this;
}

template <typename Value, typename Derivative>
template <typename Expr>
void Variable<Value, Derivative>::setExpression(
    Expression<Expr> const& expression) const
{
    mRef->setExpression(expression);
#ifndef AUTODIFF_NO_EAGER_EVALUATION
    mRef->evaluate();
#endif
}

template <typename Value, typename Derivative>
void Variable<Value, Derivative>::setDerivative(Derivative derivative) const
{
    mRef->setDerivative(std::move(derivative));
}

template <typename Value, typename Derivative>
auto Variable<Value, Derivative>::_node() const
    -> internal::AbstractComputation*
{
    return mRef.operator->();
}

template <typename Value, typename Derivative>
auto Variable<Value, Derivative>::_valueImpl() const -> Value const&
{
    return mRef->value();
}

template <typename Value, typename Derivative>
auto Variable<Value, Derivative>::_pushForwardImpl() const -> Derivative const&
{
    return mRef->derivative();
}

template <typename Value, typename Derivative>
template <typename OtherDerivative>
void Variable<Value, Derivative>::_pullBackImpl(
    OtherDerivative const& gradient) const
{
    mRef->addGradient(gradient);
}

template <typename Value, typename Derivative>
void Variable<Value, Derivative>::_transferChildrenToImpl(internal::Node& node)
{
    mRef.transferOperationTo(node);
}

// free functions
// =========================================================

/**
 * @brief Check whether two variables point to the same computation.
 */
template <typename Value, typename Derivative>
auto operator==(Variable<Value, Derivative> const& left,
    Variable<Value, Derivative> const& right) -> bool
{
    return left.mRef == right.mRef;
}

/**
 * @brief Check whether two variables point to different computations.
 */
template <typename Value, typename Derivative>
auto operator!=(Variable<Value, Derivative> const& left,
    Variable<Value, Derivative> const& right) -> bool
{
    return left.mRef != right.mRef;
}

template <typename Value, typename Derivative>
auto d(Variable<Value, Derivative> const& variable) -> Derivative const&
{
    return variable.mRef->derivative();
}

// Variable factories
// =========================================================

template <Evaluable T>
auto var(T const& literal) -> typename detail::VariableFromValue<T>::type
{
    return typename detail::VariableFromValue<T>::type(literal);
}

template <typename Expr>
auto var(Expression<Expr> const& expression) ->
    typename detail::VariableFromExpr<Expr>::type
{
    return typename detail::VariableFromExpr<Expr>::type(expression);
}

template <typename Value, typename Derivative>
auto var(Variable<Value, Derivative> const& variable)
    -> Variable<Value, Derivative>
{
    Variable<Value, Derivative> newVariable;
    newVariable.setExpression(variable);
    return newVariable;
}

} // namespace AutoDiff
