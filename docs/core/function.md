# Functions

The `AutoDiff::Function` class provides an interface to evaluate and differentiate a program.
In AutoDiff, a program is a directed acyclic graph (DAG) of expressions, where the nodes are [variables](variable.md) and the edges are dependencies between them specified by [expressions](expression.md).

```cpp
// Functions in AutoDiff

#include <AutoDiff/Core>  // for Function
#include <AutoDiff/Basic> // for Real, *, +, etc.

// Computational graph representing a program
//   z
//  ↙↘
// u  v
// ↓↘↙↓
// x  y
AutoDiff::Real x, y, u, v, z;
u = x * y; // expressions that define the program
v = x + y;
z = exp(-u / v);

// Function f(x, y) = z that evaluates the program
AutoDiff::Function f(from(x, y), to(z));

x = 1, y = 2; // set input values
f.evaluate(); // evaluate f(1, 2)
z();          // z = 0.513417

f.pullGradientAt(z); // compute gradient at (1, 2)
d(x);                // ∂f/∂x = -0.228185
d(u);                // ∂f/∂u = -0.171139
```

## Defining functions

A mathematical function definition specifies the *domain* and *codomain* of the function, as well as the *expression* that maps a point in the domain to a point in the codomain.
For instance,

$$
f \colon \mathbb{R} \times \mathbb{R} \to \mathbb{R} \ ,\ (x, y) \mapsto z = x + y
$$

defines a function $f$ that takes two real numbers $x$ and $y$ as input and returns their sum $z$ as output.

Similarly, an AutoDiff function is characterized by a set of *source* (input) variables and a set of *target* (output) variables.

```cpp
// AutoDiff function definition
#include <AutoDiff/Core>                 // for Function, from, to
#include <AutoDiff/Basic>                // for Real, +
AutoDiff::Real x, y, z;                  // floating point variables
AutoDiff::Function f(from(x, y), to(z)); // f(x, y) = z
z = x + y;                               // expression evaluation
```

The variadic template functions `from` and `to` are used to specify the source and target variables, respectively.
They can be used without namespace qualification because they are found through Argument-Dependent Lookup (ADL).

The source variables specified in the `Function` constructor are only used to limit the scope of the function.

> [!TIP]
> Sources can be omitted if they are literals, i.e., leaves in the computational graph at the time of function compilation, [see below](#function-compilation).
>
> For instance, in the example above, the function `f` could be defined without specifying the source variables `x` and `y`, since both are literals.
>
> ```cpp
> // Omitting redundant function sources
> Real x, y, z;
> Function f(z); // no sources specified
> z = x + y;     // implies x, y are sources
> ```

Sources are ignored if they are not referenced in any expression at the time of function compilation.

```cpp
// Unreachable function sources are ignored
Real x, y, z, u, v;
Function f(from(u, v), to(z));
z = x + y; // implies x, y are sources
           // u, v will be ignored
```

## Evaluating functions

While expressions are [eagerly evaluated](expression.md#eager-evaluation) as soon as they are bound to a variable, functions allow for [lazy evaluation](https://en.wikipedia.org/wiki/Lazy_evaluation) of the program.
This evaluation strategy is more efficient when the same program is evaluated multiple times with different input values, because resources can be reused.

To evaluate a program using a function `f`, first set the program input by assigning values to the source variables of `f`, then call `f.evaluate()`.
The program output and intermediate results are stored in the respective variables (see [call operator](variable.md#getters)).

```cpp
// Function evaluation
#include <AutoDiff/Core>  // for var, Function, from, to
#include <AutoDiff/Basic> // for Real, +

// Define program z = x + y
AutoDiff::Real x, y;  // variables (x = y = 0)
auto z = var(x + y);  // eagerly evaluated expression (z = 0)
// Define function f(x, y) = z
AutoDiff::Function f(z);

// Evaluate f(1, 2)
x = 1, y = 2;
f.evaluate();
z(); // z = 3

// Re-evaluate f(3, 4)
x = 3, y = 4;
f.evaluate();
z(); // z = 7
```

## Forward-mode differentiation

> In forward mode, partial derivatives are accumulated in the same order as the program is evaluated.
> Use forward mode when the number of source variables is smaller than or equal to the number of target variables.

> [!IMPORTANT]
> Ensure you evaluate a program before differentiating it. This evaluation can be done either [eagerly](expression.md#eager-evaluation) or [lazily](#evaluating-functions).

A typical use case is to compute the tangent vector to a curve $\gamma \colon \mathbb{R} \to \mathbb{R}^n$.

```cpp
// Tangent vector to a circle (forward-mode differentiation)
#include <AutoDiff/Core>  // for Function, from, to
#include <AutoDiff/Basic> // for Real, cos, sin

AutoDiff::Real t(0), x, y;
x = cos(t), y = sin(t);

AutoDiff::Function gamma(from(t), to(x, y))
gamma.pushTangentAt(t);
std::cout << "dx/dt = " << d(x); // dx/dt = 0
std::cout << "dy/dt = " << d(y); // dy/dt = 1
```

The tangent vector in the above example is a special case of the Jacobian matrix.
In general, if your program computes a function $f \colon \mathbb{R}^m \to \mathbb{R}^n$, $x \mapsto y$, then calling `pushTangentAt(x)` computes the Jacobian matrix

$$
J_f(x) = \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} \ \ldots\ \frac{\partial f_1}{\partial x_m} \\
    \vdots \\
    \frac{\partial f_n}{\partial x_1} \ \ldots\ \frac{\partial f_n}{\partial x_m}
\end{bmatrix} \in \mathbb{R}^{n \times m}
$$

and stores it in `d(y)`.

The `pushTangentAt(AbstractVariable const& seed)` member function is really a convenience function for the more general `pushTangent` function and performs the following steps:

1. Set the derivative of the source variable `seed` to the identity map
2. Set the derivative of any other source variable to zero (with appropriate dimensions)
3. Call the `pushTangent` method to compute intermediate and output derivatives.

```cpp
// Equivalent to the previous example
t.setDerivative(1); // scalar identity
f.pushTangent();
std::cout << "dx/dt = " << d(x); // dx/t = 0
std::cout << "dy/dt = " << d(y); // dy/t = 1
```

By manually setting the derivatives of all (!) source variables, you can compute any Jacobian-vector product $\delta y = J_f(x) \cdot \delta x$ (or Jacobian-derivative product) without actually computing the Jacobian matrix.

```cpp
// Directional derivative (Jacobian-vector product)
auto x = var(Eigen::Vector3d{1, 2, 3});
auto m = Eigen::MatrixXd{{1, 2, 3}, {4, 5, 6}};
auto y = var(m * x);      // matrix-vector product

Function f(y);            // f : R³ → R², x ↦ y = Mx
auto delta_x = Eigen::Vector3d{1, 1, 1};
x.setDerivative(delta_x); // set the direction vector
f.pushTangent()
std::cout << "δy =\n" << d(y) << '\n'; // δy =
                                       //  6
                                       // 15
```

## Reverse-mode differentiation (aka backpropagation)

> In reverse mode, partial derivatives are accumulated in the reverse order of evaluation.
> Use reverse mode when the number of target variables is smaller than the number of source variables.

A typical use case is to compute the gradient of a scalar function $f \colon \mathbb{R}^m \to \mathbb{R}$.

```cpp
// Gradient of the vector norm (reverse-mode differentiation)
auto x = var(Eigen::Vector3d{1, 2, 3});
auto y = var(norm(x)); //L²-norm

Function f(y);         // f : R³ → R, x ↦ y = ||x||
f.pullGradientAt(y);
std::cout << "∇f = " << d(x) << '\n'; // ∇f = 0.267261 0.534522 0.801784
```

> [!IMPORTANT]
> In AutoDiff we use the term "gradient" to refer to the derivative. See [Gradients are cotangent vectors](../math/diff-geo.md#gradients-are-cotangent-vectors) for a mathematical discussion.

## Function compilation

Note that a `Function` can be instantiated without specifying the expressions that map the source variables to the target variables.

> [!NOTE]
> The program to be evaluated by a function is defined by the expressions at the time of function compilation.

If necessary, compilation is triggered by the first (lazy) evaluation or differentiation of a function.
Compilation creates a non-owning view of the computation graph, which is used to efficiently evaluate or differentiate a function.

```cpp
// Function compilation on first evaluation
#include <AutoDiff/Core>  // for var, Function, to
#include <AutoDiff/Basic> // for Real, +, *

AutoDiff::Real x, y;      // floating point variables
auto z = var(x + y);      // expression evaluation
AutoDiff::Function f(z);  // f(x, y) = z

x = 1, y = 2;  // set input values
f.compiled();  // false
f.evaluate();  // first evaluation, triggers compilation
f.compiled();  // true
z();           // 3.0

y = 3;
f.evaluate();  // re-evaluation, no recompilation
z();           // 4.0
```

> [!CAUTION]
> After compiling a function `f`, you can still modify its program through [expression assignment](expression.md#when-assigning-to-existing-variables).
> However, if you change the expression of a non-source variable, you must then explicitly recompile the function by calling `f.compile()`.
> Otherwise, the program might crash or produce incorrect results.

```cpp
// ...continuing from the previous example
z = x * y;    // expression assignment to non-source variable, invalidates f
f.compile();  // must recompile f
f.evaluate(); // re-evaluation, no recompilation
z();          // 3.0
```

You can also call `f.compile()` before the first evaluation or differentiation to avoid the (small) overhead of compiling the program then.

```cpp
Function f(z);
f.compile();
f.compiled(); // true
f.evaluate(); // no compilation needed
```

## Next steps

For a complete guide, see the [Documentation](../index.md).
