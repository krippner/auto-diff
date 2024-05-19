# Expressions

An expression in AutoDiff is a combination of one ore more AutoDiff variables, other expressions, and literals (values) using operators or function calls.

```cpp
// Examples of expressions
AutoDiff::Variable<double, double> x; // a variable is itself an expression
auto expr1 = x + 1;                   // variable + literal
auto expr2 = 2 * expr1;               // literal * expression
auto expr3 = sin(expr2);              // function call
auto expr4 = log(x * expr3) - 1;      // nested expressions
```

> [!TIP]
> Always use the `auto` keyword to let the compiler deduce the type of an expression.
> These types can be long and unwieldy because of [expression templates](https://en.wikipedia.org/wiki/Expression_templates), which build entire expression trees at compile time in order to generate optimized code to evaluate them.

AutoDiff [modules](../index.md#modules) provide overloaded operators and functions to create expressions for supported value and derivative types.
Additionally, you can define your own [custom expressions](#custom-expressions).

> [!IMPORTANT]
> All variables in an expression must have the same derivative type.
>
> ```cpp
> // Derivative type mismatch
> AutoDiff::Variable<double, double> x;
> AutoDiff::Variable<double, float>  y;
> auto expr = x + y; // STATIC ASSERTION FAILURE:
>                    // operands must have the same derivative type
> ```

## Evaluating expressions

While expressions provide the instructions for evaluation and differentiation, [variables](variable.md#top) must be used to actually compute and cache values and derivatives.

### During variable creation

In the same way you can [create variables from values](variable.md#creating-variables-from-values) using the `var` factory function and the `Variable` constructor, you can also create variables that evaluate expressions.

```cpp
// Evaluate expression to new variable
#include <AutoDiff/Core>
#include <AutoDiff/Basic> // for Real, *
using AutoDiff::Real;

Real x(2), y(3);
auto expr = x * y;   // Basic::Product<Real, Real> (expression template)

auto z1 = var(expr); // factory function returns Real
z1();                // 6, (eagerly) evaluated expression

Real z2(expr);       // Real constructor
z2();                // 6, (eagerly) evaluated expression
```

Note that the return type of `var` is automatically deduced by the compiler based on the expression template and the included module (e.g., `AutoDiff/Basic`).
Also, it is not necessary to prepend the namespace because `var` is found through Argument-Dependent Lookup (ADL) when used with expressions.

### When assigning to existing variables

The expression evaluated by a variable can be dynamically changed by assigning a new expression to the variable.
Assigning a literal to a variable removes the expression (if any) and sets the value to the literal.

```cpp
// Evaluate expression to existing variable
#include <AutoDiff/Core>
#include <AutoDiff/Basic> // for Real, *, +

AutoDiff::Real x(2), y(3), z(0);

z = x * y; // assign expression to variable
z();       // 6, (eagerly) evaluated expression

z = x + y; // assign new expression
z();       // 5, (eagerly) evaluated expression

z = 3;     // assign literal, removes expression
z();       // 3, literal value
```

> [!CAUTION]
> Do not assign expressions that contain the variable itself, because this creates a circular dependency.
> It will compile, but deferred evaluation or differentiation (see [Functions](function.md#top)) will throw a runtime exception.
>
> ```cpp
> // Circular dependency
> Real x(2), y(3);
> x = x + y;      // BAD:  this is not what you want
> x = var(x + y); // GOOD: create a new variable and point x to it
> ```

### Eager evaluation

By default, expressions are evaluated eagerly, meaning that the value is computed immediately when the variable is created or the expression assigned.
This is the most intuitive behavior, just like in regular C++ code, and is especially useful for debugging and testing, because you can inspect the value of an expression and locate errors more easily.

However, sometimes you may want to [defer the evaluation](function.md#evaluating-functions) of an expression, for example, when the value of variables is not known at the time of their creation.
In this case, eager evaluation can be disabled by defining the `AUTODIFF_NO_EAGER_EVALUATION` macro *before* including the framework header.

```cpp
// To disable eager evaluation, define the following
// macro **before** including the framework header.
#define AUTODIFF_NO_EAGER_EVALUATION
#include <AutoDiff/Core>  // framework header
#include <AutoDiff/Basic>
using AutoDiff::Real;

Real x(2), y(3);
Real z(x * y); // expression is not evaluated yet
z();           // z has default-constructed value 0
```

## Custom expressions

What all expressions have in common is that they are subclasses of the `AutoDiff::Expression<Derived>` class template, which defines the static interface for the evaluation and differentiation of expressions.
The `Derived` template argument is the type of the derived class, which inherits the static interface through the [Curiously Recurring Template Pattern (CRTP)](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) and implements the actual evaluation and differentiation logic.

If your custom expression is a composition of existing expressions, you don't need to implement the `Operation` interface yourself.
Instead, you can wrap the expression in a function that returns the expression, as in the following example.

```cpp
// Example of a composite expression
#include <AutoDiff/Core> // for Expression

template <typename Derived>
auto sigmoid(Expression<Derived> const& input)
{
    return 1 / (1 + exp(-4 * input));
}
```

By accepting an `Expression` base class reference, the function can be called with any expression.

```cpp
// Using the composite expression
#include <AutoDiff/Basic>
using AutoDiff::Real;

Real x(0);
auto expr1 = sigmoid(x);
Real y1(expr1);
y1(); // 0.5

auto expr2 = sigmoid(expr1);
Real y2(expr2);
y2(); // 0.880797
```

To see how you can implement custom expressions that are not compositions, please refer to the [developer guide](../developer.md).

## Expressions vs. variables

During gradient computation, derivatives are accumulated in reverse order of evaluation.
To speed up computation, the results from the evaluation pass need be stored in memory.
Doing so for every operation, however, could lead to excessive memory usage.
Expressions help mitigate this issue by not storing any value, significantly reducing the memory footprint.

Another advantage of expressions is that they can be optimized by the compiler to choose efficient algorithms and reduce the number of temporary variables.
The Eigen linear algebra library, for example, makes [heavy use of expression templates](https://eigen.tuxfamily.org/dox/TopicInsideEigenExample.html) to produce optimized code.

As a rule of thumb, evaluate an expression to a variable if

- it is used in multiple expressions,

    ```cpp
    // evaluates x only once
    auto x = var(..);
    u = x + 2;
    v = x * 3;
    ```

- it is updated in a loop,

    ```cpp
    auto x = var(..);
    for (int i = 0; i < 10; ++i) {
        x = var(x + 1); // evaluate to a NEW variable
    }
    ```

- or you need to access its value or derivative.

Otherwise, use benchmarks to see whether introducing a variable leads to a significant speedup.
Giving you control over this space-time trade-off is a key aspect of the API design.

## Memory management in expressions

In AutoDiff, variables handle their own resource, such as their value and derivative, using the [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization) idiom.
This means that the resource is acquired when the variable is created and released when the variable is destroyed.

Through an expression, a variable can depend on other variables (operands), which in turn can depend on other variables, and so on.
As long as a variable is bound to at least one expression, its resource cannot be deallocated, even if it is no longer referenced in the code.
Once the last expression referencing a variable is destroyed, the variable and its resource is automatically garbage-collected.

This is contrary to tape-based automatic differentiation, where memory is managed by a global data structure called the "tape".
The tape records each operation as it is performed during the computation and stores the intermediate results.

> [!NOTE]
> Safe deallocation: dependent variables are always destructed *iteratively*, preventing stack overflow due to deep recursion.

The loop in the following code creates a large number of dependent variables that are not deallocated until the end of the scope, because they are still referenced indirectly by the `state` variable.

```cpp
// Safe destruction of dependent variables
Real initial(0);
{
  Real state = initial;
  for (int i = 0; i < 10000; ++i) {
    state = var(state + 1); // state is a new variable each iteration
    // previous state is NOT deallocated
    // new state depends on it through an expression
  }
  // state now depends on 10000 variables
} // end of scope: state is destructed and with it
  // all referenced variables (except 'initial') are safely deallocated
```

## Next steps

Next up: [Functions](function.md).

For a complete guide, see the [Documentation](../index.md).
