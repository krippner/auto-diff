# Eigen module

To use the Eigen module, include the following header:

```cpp
#include <AutoDiff/Eigen>
```

## Supported types

### Aliases

For your convenience, the Eigen module provides the following type aliases:

```cpp
// Type aliases provided by the Eigen module

namespace AutoDiff {

// scalar types
using Real    = Variable<double, Eigen::MatrixXd>;
using Integer = Variable<int, Eigen::MatrixXd>;
using Boolean = Variable<bool, Eigen::MatrixXd>;

using RealF    = Variable<float, Eigen::MatrixXf>;
using IntegerF = Variable<int, Eigen::MatrixXf>;
using BooleanF = Variable<bool, Eigen::MatrixXf>;

// vector and dense matrix types
using Vector   = Variable<Eigen::VectorXd, Eigen::MatrixXd>;
using Vector2d = Variable<Eigen::Vector2d, Eigen::MatrixXd>;
using Vector3d = Variable<Eigen::Vector3d, Eigen::MatrixXd>;
using Vector4d = Variable<Eigen::Vector4d, Eigen::MatrixXd>;
using Matrix   = Variable<Eigen::MatrixXd, Eigen::MatrixXd>;
using Matrix2d = Variable<Eigen::Matrix2d, Eigen::MatrixXd>;
using Matrix3d = Variable<Eigen::Matrix3d, Eigen::MatrixXd>;
using Matrix4d = Variable<Eigen::Matrix4d, Eigen::MatrixXd>;

using VectorXf = Variable<Eigen::VectorXf, Eigen::MatrixXf>;
using Vector2f = Variable<Eigen::Vector2f, Eigen::MatrixXf>;
using Vector3f = Variable<Eigen::Vector3f, Eigen::MatrixXf>;
using Vector4f = Variable<Eigen::Vector4f, Eigen::MatrixXf>;
using MatrixXf = Variable<Eigen::MatrixXf, Eigen::MatrixXf>;
using Matrix2f = Variable<Eigen::Matrix2f, Eigen::MatrixXf>;
using Matrix3f = Variable<Eigen::Matrix3f, Eigen::MatrixXf>;
using Matrix4f = Variable<Eigen::Matrix4f, Eigen::MatrixXf>;

// array types
using Array   = Variable<Eigen::ArrayXd, Eigen::ArrayXd>;
using ArrayXX = Variable<Eigen::ArrayXXd, Eigen::ArrayXXd>;

using ArrayXf  = Variable<Eigen::ArrayXf, Eigen::ArrayXf>;
using ArrayXXf = Variable<Eigen::ArrayXXf, Eigen::ArrayXXf>;

} // namespace AutoDiff
```

## Operations

Generally, one of the expressions in binary operations can be replaced by a literal of the same type.

```cpp
auto x = var(Eigen::Vector3d{1, 2, 3}); // vector variable
dot(x, Eigen::Vector3d{4, 5, 6});       // dot product with vector literal
```

The following operations are currently supported.

### Scalar operations

The Eigen module supports the same operations as the [Basic module](basic.md), but with Eigen derivatives.

### Array and element-wise vector/matrix operations

In array and element-wise matrix operations, scalar literals and expressions are also accepted and broadcasted to the shape of the array or matrix.

```cpp
auto x = var(Eigen::Matrix2d{{1, 2}, {3, 4}}); // matrix variable
x + 5; // add 5 to each element
```

- `+`, `-`, `*`, `/`: Element-wise arithmetic operations.
- `pow`: Power function, element-wise.
- `sin`: Sine function, element-wise.
- `cos`: Cosine function, element-wise.
- `exp`: Exponential function, element-wise.
- `log`: Natural logarithm, element-wise.
- `sqrt`: Square root, element-wise.
- `square`: Square function, element-wise.
- `min`: Element-wise minimum of an expression and zero.
- `max`: Element-wise maximum of an expression and zero.

### Matrix products

- `dot`: Dot product of two vectors.
- `tensorProduct`: Tensor product of two vectors.
- `*`: Matrix-matrix and matrix-vector products.

### Matrix reductions

- `total`: Sum of matrix elements.
- `mean`: Mean of matrix elements.
- `norm`: Frobenius ($L^2$) norm of a matrix.
- `squaredNorm`: Squared Frobenius ($L^2$) norm of a matrix.

## Matrix-valued expressions

During differentiation, AutoDiff flattens matrix expressions in column-major order.
This ensures that the derivative (Jacobian matrix) is always an `Eigen::Matrix`.

```cpp
auto m1 = Eigen::MatrixXd{{1, 2}, {3, 4}};
auto m2 = Eigen::MatrixXd{{5, 6, 7}, {8, 9, 10}};
auto x = var(m1);    // 2⨉2 matrix variable
auto y = var(m2);    // 2⨉3 matrix variable
auto u = var(x * y); // 2⨉3 matrix variable
Function f(u);
f.pullGradientAt(u);
d(u);                // 6⨉6 identity matrix
d(x);                // 6⨉4 matrix
d(y);                // 6⨉6 matrix
```
