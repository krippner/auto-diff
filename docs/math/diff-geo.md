# Automatic differentiation and differential geometry

Automatic differentiation (AD) has a particularly elegant description in the language of [differential geometry](https://en.wikipedia.org/wiki/Differential_geometry), which generalizes calculus.
The design of the _AutoDiff_ library honors that connection.
For instance, the _Core_ framework of _AutoDiff_ implements the abstract coordinate-free concepts of differential, tangent vector, and gradient.
The _Basic_ and _Eigen_ modules then provide concrete implementations of these concepts in terms of floating-point numbers and arrays.

For the interested reader, this article very briefly summarizes the mathematical background of AD and provides links to some in-depth resources.
To start, the article goes over the definitions of [smooth manifolds](#smooth-manifolds) and [tangent vectors](#tangent-vectors-on-manifolds), which are central to the concept of [differentiation](#differentiation) in differential geometry.
The article concludes with the statement that [forward- and reverse-mode AD](#forward--and-reverse-mode-ad) act on tangent vectors and gradients, respectively.

## Smooth manifolds

A [smooth manifold](https://en.wikipedia.org/wiki/Differentiable_manifold) $M$ of dimension $m$ is a generalization of Euclidean space $\mathbb{R}^m$ to a topological space that _only locally_ resembles $\mathbb{R}^m$.
More precisely, around every point $p \in M$ there exists a [neighborhood](https://en.wikipedia.org/wiki/Open_set#Topological_space) $U \subseteq M$ and a [smooth bijection](https://en.wikipedia.org/wiki/Diffeomorphism) $x \colon U \to \mathbb{R}^m$.

The pair $(U,x)$ is called a [coordinate chart](https://en.wikipedia.org/wiki/Topological_manifold#Coordinate_charts) of the manifold.
Using charts, abstract objects like tangent vectors, gradients, and differentials can be represented by arrays of real coefficients such as column vectors, row vectors, and matrices.
This is useful for concrete calculations.
For details please see [Differentiation in local coordinates](diff-geo-chart.md).

## Tangent vectors on manifolds

The notion of a [tangent vector](https://ncatlab.org/nlab/show/tangent+bundle) $v$ at $p$ generalizes the directional derivative of smooth functions over $\mathbb{R}^m$ to a derivative along smooth curves on $M$.

Concretely, a tangent vector $v$ is an equivalence class $[\gamma]_{\sim}$ of smooth curves $\gamma \colon \mathbb{R} \to M$ that pass through $p$ at parameter $0$, i.e., $\gamma(0) = p$.
Two such curves $\gamma_1$ and $\gamma_2$ are considered equivalent if their derivatives agree in any chart $x$,

$$
\gamma_1 \sim \gamma_2 \Leftrightarrow  (x \circ \gamma_1)'(0) = (x \circ \gamma_2)'(0) .
$$

The composition $x \circ \gamma \colon \mathbb{R} \to \mathbb{R}^m$, $t \mapsto x(\gamma(t))$ is the representation of the curve $\gamma$ in local coordinates $x$.
In general, the derivative of a smooth function $f$ along any curve $\gamma \in [\gamma]_{\sim}$ is the tangent vector

$$
v \colon C^{\infty}(U) \to \mathbb{R}, \quad f \mapsto (f \circ \gamma)'(0) .
$$

### Tangent space

The collection of all tangent vectors at $p \in M$ forms an $m$-dimensional vector space, called the [tangent space](https://en.wikipedia.org/wiki/Tangent_space) $T_pM$ at $p$.

The disjoint union of all tangent spaces on $M$ is called the [tangent bundle](https://en.wikipedia.org/wiki/Tangent_bundle) $TM$ over $M$ and consists of ordered pairs $(p,v)$ where $p \in M$ and $v \in T_pM$.

## Differentiation

Let $\phi \colon M \to N$ be a smooth function between smooth manifolds $M$ and $N$.

Then there is an associated map $\phi^* \colon C^{\infty}(N) \to C^{\infty}(M)$ defined by the precomposition $f \mapsto f \circ \phi$.
The function $\phi^*(f)$ is called the [pullback](https://en.wikipedia.org/wiki/Pullback) of $f$ by $\phi$.

The _differentiation_ or _derivative operator_ ${\rm d}$ sends the function $\phi$ to its _differential_ or _derivative_

$$
{\rm d}\phi \colon TM \to TN, \quad (p,v) \mapsto (\phi(p),v \circ \phi^*) .
$$

That is, for each point $p \in M$ there is a _linear_ map

$$
{\rm d}\phi_p \colon T_pM \to T_{\phi(p)}N, \quad v \mapsto v \circ \phi^* := \phi_{*p}(v).
$$

The tangent vector $\phi_{*p}(v)$ is called the [pushforward](https://en.wikipedia.org/wiki/Pushforward_(differential)) of $v$ by $\phi$.

## Forward- and reverse-mode AD

In the context of automatic differentiation, the _forward-mode_ and _reverse-mode_ algorithms act on tangent vectors and cotangent vectors (aka gradients), respectively.

### Tangent vectors are differentials

Given a chart $(U,x)$ on a smooth manifold $M$, the differential at $p \in U$

$$
\tag{1}
{\rm d}x_p \colon T_pM \to \mathbb{R}^m, \quad v \mapsto v(x) = {\rm d}(x \circ \gamma)_0
$$

is a linear isomorphism between the tangent space $T_pM$ and $\mathbb{R}^m$ (see Theorem [here](https://en.wikipedia.org/wiki/Tangent_space#The_derivative_of_a_map)).

This means that in a chart $x$, every tangent vector $v$ has a unique representation as the differential of the curve $x \circ \gamma$ at $0$.

### Gradients are cotangent vectors

Given a smooth function $\phi \colon M \to \mathbb{R}^n$ and a chart $(\phi(U), y)$ on $\mathbb{R}^n$, we have for every point $p \in U$ and tangent vector $v \in T_pM$ that

$$
({\rm d}\phi_p)(v)(y) = v(y \circ \phi) = v(y) \circ \phi \mapsto v(\phi) \in \mathbb{R}^n \ .
$$

The last map is by the inverse of the linear isomorphism (1).

Taking $n=1$, this means that for a scalar function $f \colon M \to \mathbb{R}$ and $p \in M$, the gradient ${\rm d}f_p$ is a linear functional on $T_pM$ that maps tangent vectors to $\mathbb{R}$.
In other words, the gradient is a cotangent vector in $T_p^*M = (T_pM)^*$.

> [!IMPORTANT]
> Usually, the [gradient](https://en.wikipedia.org/wiki/Gradient) $\nabla f(p)$ is defined as the unique vector such that $\langle \nabla f(p), v \rangle = {\rm d}f_p(v)$ for all tangent vectors $v \in T_pM$.
>
> 1. This definition of the gradient is **non-canonical** because it requires an extra inner product $\langle \cdot,\cdot \rangle$ on the tangent space $T_pM$.
> 2. Vectors are pushed forward by the derivative, while covectors are pulled back.
> A "gradient vector" cannot be pulled back using backpropagation (without an inner product).
>
> The term "gradient" appears in the AutoDiff API due to its frequent use in automatic differentiation. However, for mathematical consistency, we use "gradient" to refer to the differential, which is a covector, not a vector.

To be continued...

## References

- [nLab - tangent bundle](https://ncatlab.org/nlab/show/tangent+bundle)
- [Wikipdia - Differentiable manifold](https://en.wikipedia.org/wiki/Differentiable_manifold)
