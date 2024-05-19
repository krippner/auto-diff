# Mathematical notes on the API

> [!TIP]
> For more detailed mathematical background on differentiation, see [Automatic differentiation and differential geometry](diff-geo.md).

The `Variable` class in AutoDiff uses mathematical function notation for the getters of value and derivative.

```cpp
Real x; // create a variable
x();    // get the value of x
d(x);   // get the derivative of x
```

Indeed, depending on the current evaluation mode (forward or reverse), a variable in AutoDiff can be interpreted precisely as a particular mathematical function.

## Forward mode

Forward mode is used when you call `evaluate` or `pushTangent` on a `Function` object.

### Literal variables

In this context, a _literal variable_

```cpp
AutoDiff::Vector3d x;
```

is a smooth function

$$
x \colon M \to \mathbb{R}^3 \ ,\ p \mapsto x(p)
$$

on a smooth manifold $M$.

By setting the value of the variable

```cpp
x = Eigen::Vector3d{1, 2, 3};
```

we are setting $p$ such that $x(p) = (1, 2, 3)$.
Since we never need to refer to $p$ directly, we can omit it from the notation.

```cpp
x(); // returns x(p) = (1, 2, 3)
```

Because $x$ is a smooth function, it has a derivative $dx_p$ at the point $p$.
In general, the derivative is a linear map and can be represented by a Jacobian matrix.
For instance, we can set $dx_p$ to the identity matrix.

```cpp
x.setDerivative(Eigen::Matrix3d::Identity()); // set the Jacobian matrix
```

In the case where $M = \mathbb{R}$, the function $x$ is a curve in $\mathbb{R}^3$ parameterized by $p$.
The derivative $dx_p$ is then a tangent vector to the curve at $x(p)$.

```cpp
x.setDerivative(Eigen::Vector3d{1, 1, 1}); // set the tangent vector
```

In differential geometry, the derivative (also called _differential_) is sometimes written as ${\rm d}(x)_p$ which emphasizes that it can be constructed by applying the [_exterior derivative_](https://en.wikipedia.org/wiki/Exterior_derivative) ${\rm d}$ to the function $x$.
Similarly, the AutoDiff `d` operator returns the differential of a variable evaluated at $p$.

```cpp
d(x); // returns d(x)_p = (1, 1, 1)^T
```

### Expression variables

More complex functions can be formed by defining a variable as the expression of other variables.

```cpp
AutoDiff::Vector2d y = phi(x); // expression variable
```

The _expression variable_ `y` represents the composition of two functions,

$$
y = \phi\ \circ\ x \colon M \to \mathbb{R}^2,\ p \mapsto \phi(x(p))\ .
$$

Like before, the variable `y` can be evaluated as function, where $p$ is omitted.

```cpp
y(); // returns y(p) = ϕ(x(p)) = ϕ((1, 2, 3))
```

Automatic differentiation builds on the [chain rule](https://en.wikipedia.org/wiki/Chain_rule), which states that the differential of a composite is the composite of the differentials,

$$
dy_p = d(\phi \circ x)_p = d\phi _{x(p)} \circ dx_p\ .
$$

In AutoDiff, calling `pushTangent` on a `Function` instance applies the chain rule and computes the differential of the variable `y` by composing the derivatives from right to left.

```cpp
AutoDiff::Function f(from(x), to(y))
f.pushTangent(); // applies the chain rule
d(x);            // returns d(x)_p
d(y);            // returns d(y)_p = d(ϕ)_{x(p)} ∘ d(x)_p
```

## Reverse mode

Reverse mode is used when you call `pullGradient` on a `Function` object.

In reverse mode, the same variables are interpreted as different mathematical functions.
The output variable `y` is now the function

$$
\hat{y} \colon \mathbb{R}^2 \to N \ ,\ y(p) \mapsto q
$$

mapping the value $y(p)$ from the forward pass into a smooth manifold $N$.

In the case where $N = \mathbb{R}$, the derivative $d\hat{y}$ is a cotangent vector (aka gradient) at $y(p)$ and can be represented by a row vector.

```cpp
y.setDerivative(Eigen::RowVector2d{1, 1}); // set the cotangent vector
d(y); // returns d(ŷ) = (1, 1)
```

The variable `x` is now associated with the composition

$$
\hat{x} = \hat{y} \circ \phi \colon \mathbb{R}^3 \to N \ ,\ x(p) \mapsto q
$$

with $x(p)$ being the input value from the forward pass.

Reverse-mode automatic differentiation uses again the chain rule to compute the gradient of $\hat{y}$ with respect to $x$.

$$
d\hat{x}_x = d\hat{y}_y \circ d\phi_x
$$

When calling `pullGradient` on a `Function` instance, the chain rule is applied and the gradients are computed by composing the derivatives from left to right, i.e., in the reverse order of evaluation.

```cpp
f.pullGradient(y); // applies the chain rule
d(x);              // returns d(x^)_{x(p)}
```

> [!TIP]
> The discussion above also extends to binary expressions.
> In this case, the domain is the Cartesian product of the domains of the two inputs.
>
> $$
> \psi \colon M_1 \times M_2 \to N \ ,\ (p_1, p_2) \mapsto q
> $$
>
> The derivative $d\psi$ at $(p_1,p_2)$ is a linear map between the corresponding tangent spaces.
> One can show that the tangent space at $(p_1,p_2)$ is also given by the Cartesian product.
>
> $$
> d\psi_{(p_1,p_2)} \colon T_{p_1}M_1 \times T_{p_2}M_2 \to T_qN \ , (v_1, v_2) \mapsto w
> $$
