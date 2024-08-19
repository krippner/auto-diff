# Basic module

To use the Basic module, include the following header:

```cpp
#include <AutoDiff/Basic>
```

## Supported types

The Basic module supports variables and operations with all value and derivative types `T` for which `std::is_arithmetic_v<T>` is `true`.
Remember, the derivative type must be the same for all variables in an expression.

### Aliases

For your convenience, the Basic module provides the following type aliases:

```cpp
// Type aliases provided by the Basic module

namespace AutoDiff {

using Real    = Variable<double, double>;
using Integer = Variable<int, double>;
using Boolean = Variable<bool, double>;

using RealF    = Variable<float, float>;
using IntegerF = Variable<int, float>;
using BooleanF = Variable<bool, float>;

} // namespace AutoDiff
```

## Operations

In binary operations, one of the operands can also be a scalar literal.

```cpp
Real x(2);
x + 3; // right-hand side literal
3 * x; // left-hand side literal
```

The following operations are currently supported:

- `+`, `-`, `*`, `/`: Arithmetic operations.
- `sin`, `cos`, `tan`, `cot`: Trigonometric functions.
- `asin`, `acos`, `atan`, `acot`: Inverse trigonometric functions.
- `sinh`, `cosh`, `tanh`: Hyperbolic functions.
- `exp`: Exponential function.
- `log`: Natural logarithm.
- `pow`: Power function.
- `square`: Square function.
- `sqrt`: Square root.
- `min`, `max`: Minimum, maximum of a scalar expression and zero.
