# Applications

This section provides a list of examples demonstrating the use AutoDiff in common scenarios.
The examples assume that you imported the following headers and symbols:

```cpp
#include <AutoDiff/Core>
#include <AutoDiff/Eigen>
#include <Eigen/Core>
#include <iostream>

using AutoDiff::var, AutoDiff::Real, AutoDiff::Function, AutoDiff::Expression;
using std::cout;
```

You can also find these examples in the file [examples/eigen.cpp](../examples/eigen.cpp).

## Control flow

### Function calls

```cpp
// This function can take any AutoDiff expression as input
// and returns the sigmoid function applied element-wise.
template <typename X>
auto logistic(Expression<X> const& x) {
    return 1 / (1 + exp(-4 * x)); // scalars broadcast to arrays
}

// passing a scalar variable
auto x1 = var(0);
auto y1 = var(logistic(x1));
cout << "y1 = " << y1() << '\n'; // y1 = 0.5

// passing an array variable
auto x2 = var(Eigen::Array3d{-1, 0, 1}.transpose());
auto y2 = var(logistic(x2));
cout << "y2 = " << y2() << '\n'; // y2 = 0.0179862       0.5  0.982014

// passing a vector expression
auto x3 = var(Eigen::RowVector3d{-1, 0, 1});
auto y3 = var(logistic(x3 / 2));
cout << "y3 = " << y3() << '\n'; // y3 = 0.119203      0.5 0.880797
```

### Loops

```cpp
// Loop example
auto initial = var(0);
auto state   = initial;
for (int i = 0; i < 10; ++i) {
    state = var(state + 1); // evaluate to a NEW variable
}

cout << "state = " << state() << '\n'; // state = 10.0

Function f(state);
f.pullGradientAt(state);
cout << "∂state/∂initial = " << d(initial)
        << '\n'; // ∂state/∂initial = 1
```

For more details on when to use `var`, see [Variables vs expressions](core/expression.md#variables-vs-expressions).

### Branches

```cpp
// Caution: if statements are not differentiable
template <typename X>
auto bad_ReLU(Expression<X> const& x) {
    return x > 0 ? x : 0; // cannot differentiate this
}
// Branches can only be differentiated through special functions
template <typename X>
auto reLU(Expression<X> const& x) {
    return max(x);
}

auto x = var(Eigen::Array3d{-1, 0, 1}.transpose());
auto y = var(reLU(x));
cout << "y = " << y() << '\n'; // y = 0 0 1
```

You can, however, use conditionals to decide which expression to evaluate:

```cpp
// Caution: if statements cannot depend on variables
Real x(1), y;
if (x() > 0) { // true
    y = var(x);
} else {
    y = var(0.0); // never evaluated!
}
// from now on y = var(x)

Function f(y);
x = -1;
f.evaluate();
cout << "y = " << y() << '\n'; // y = -1
```

## Computing the Jacobian matrix

Given a function $f \colon \mathbb{R}^m \to \mathbb{R}^n$, $x \mapsto y$, the Jacobian matrix $J_f(x) \in \mathbb{R}^{n \times m}$ is defined as

$$
J_f(x) = \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} \ \ldots\ \frac{\partial f_1}{\partial x_m} \\
    \vdots \\
    \frac{\partial f_n}{\partial x_1} \ \ldots\ \frac{\partial f_n}{\partial x_m}
\end{bmatrix} .
$$

```cpp
// Computing the Jacobian matrix
auto x = var(Eigen::Vector3d{1, 2, 3});
auto m = var(Eigen::MatrixXd{{1, 2, 3}, {4, 5, 6}});
auto y = var(m * x); // matrix-vector product

Function f(y);
f.pullGradientAt(y);
cout << "∂f/∂x =\n"
        << d(x) << '\n'; // ∂f/∂x =
                        // 1 2 3
                        // 4 5 6
cout << "∂f/∂m =\n"
        << d(m) << '\n'; // ∂f/∂m =
                        // 1 0 2 0 3 0
                        // 0 1 0 2 0 3
```

> [!NOTE]
> During differentiation, matrices are flattened column-wise.
> Therefore, `d(m)` returns a $2 \times 6$ Jacobian matrix instead of a $2 \times 2 \times 3$ tensor.
> For more details, see [Matrix-valued expressions](modules/eigen.md#matrix-valued-expressions).

For more details on the `pullGradientAt` method, see [Reverse-mode differentiation (aka backpropagation)](core/function.md#reverse-mode-differentiation-aka-backpropagation).

## Gradient computation

Given a scalar function $f \colon \mathbb{R}^m \to \mathbb{R}$, $x \mapsto y$, the gradient $\nabla f(x) \in \mathbb{R}^m$ is defined as

$$
\nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots \right] .
$$

```cpp
// Gradient computation
auto x = var(Eigen::Vector3d{1, 2, 3});
auto y = var(norm(x)); // L²-norm

Function f(y); // f : R³ → R, x ↦ y = ||x||
f.pullGradientAt(y);
cout << "∇f = " << d(x) << '\n'; // ∇f = 0.267261 0.534522 0.801784
```

Note that the gradient of a scalar function is a $1 \times n$ Jacobian matrix (aka "row vector").

For more details, see [Reverse-mode differentiation (aka backpropagation)](core/function.md#reverse-mode-differentiation-aka-backpropagation).

## Element-wise gradient computation

```cpp
// Element-wise gradient computation (array variables)
auto x = var(Eigen::Array3d{1, 2, 3}.transpose());
auto y = var(Eigen::Array3d{4, 5, 6}.transpose());
auto z = var(x * y);

Function f(z);
f.pullGradientAt(z);
cout << "∇_x f = " << d(x) << '\n'; // ∇_x f = 4 5 6
cout << "∇_y f = " << d(y) << '\n'; // ∇_y f = 1 2 3
```

If you used vector variables for `x` and `y` (as in the previous example), you would instead get $3 \times 3$ diagonal matrices for $\nabla_x f$ and $\nabla_y f$ (with the same values).

Another way to compute the element-wise gradient but with vector variables is to seed the backpropagation manually:

```cpp
// Element-wise gradient computation (vector variables)
auto x = var(Eigen::Vector3d{1, 2, 3});
auto y = var(Eigen::Vector3d{4, 5, 6});
auto z = var(cwiseProduct(x, y));

Function f(z);
z.setDerivative(Eigen::RowVector3d::Ones());
f.pullGradient();
cout << "∇_x f = " << d(x) << '\n'; // ∇_x f = 4 5 6
cout << "∇_y f = " << d(y) << '\n'; // ∇_y f = 1 2 3
```

## Jacobian-vector products

Given a function $f \colon \mathbb{R}^m \to \mathbb{R}^n$, $x \mapsto y$.
If you are only interested in the product of the Jacobian matrix $J_f(x)$ with a given direction (tangent vector) $\delta x$,

$$
\delta y = J_f \cdot \delta x \ ,
$$

then the following code using the `push_tangent` method is much more efficient than computing the full Jacobian matrix first and then multiplying it with $\delta x$.

```cpp
// Directional derivative (Jacobian-vector product)
auto x = var(Eigen::Vector3d{1, 2, 3});
auto m = Eigen::MatrixXd{{1, 2, 3}, {4, 5, 6}};
auto y = var(m * x); // matrix-vector product

Function f(y); // f : R³ → R², x ↦ y = Mx
auto delta_x = Eigen::Vector3d{1, 1, 1};
x.setDerivative(delta_x); // set direction vector
f.pushTangent();
cout << "δy =\n"
        << d(y) << '\n'; // δy =
                        //  6
                        // 15
```

For more details, see [Forward-mode differentiation](core/function.md#forward-mode-differentiation).
