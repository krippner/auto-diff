# Differentiation in local coordinates

> [!TIP]
> Here, we follow the [convention in physics](https://en.wikipedia.org/wiki/Covariance_and_contravariance_of_vectors) to label vector components by upper indices and the basis by lower indices.

## Charts

Given a smooth manifold $M$ of dimension $m$, there is a smooth bijection $x \colon U \to \mathbb{R}^m$ defined on a neighborhood $U \subseteq M$.

The pair $(U,x)$ is called a _chart_ of the manifold and assigns a set of _coordinates_ $x^i(p)$ to points $p \in U$.
The _component functions_ $x^i \colon U \to \mathbb{R}$ project to the $i$th component, $p \mapsto {\rm proj}_i(x(p))$.

## Tangent vectors in charts

A tangent vector at $p \in M$ is a linear map $v \colon C^{\infty}(M) \to \mathbb{R}$
(see definition in [Tangent vectors on manifolds](diff-geo.md#tangent-vectors-on-manifolds)).

The tangent space $T_pM$ at $p$ is an $m$-dimensional vector space.
This means that $T_pM$ has a basis of $m$ tangent vectors, usually denoted $\left( \frac{\partial}{\partial x^i} \right)_p$ , which depends on the choice of coordinates $x^i$ such that

$$
\left( \frac{\partial}{\partial x^i} \right)_p(x^j) = \delta _{ij} .
$$

This also means that every tangent vector $v \in T_pM$ has a unique expansion

$$
v = \sum_{i=1}^m v^i \left( \frac{\partial}{\partial x^i} \right)_p
$$

with coefficients $v^i \in \mathbb{R}$.
Since

$$
v(x^j) = \sum _{i=1}^m v^i \left( \frac{\partial}{\partial x^i} \right)_p(x^j) = v^j ,
$$

the representation of a tangent vector $v$ in the chart $x$ is the column vector

$$
v(x) = \begin{bmatrix} v^1 \\ \vdots \\ v^m \end{bmatrix}.
$$

## Differentials in charts

To be continued...
