# Variables

Variables are the basic building blocks of the AutoDiff framework.
They represent persistent, shared state in your program and store input values, intermediate results, and output values.
Additionally, variables are used to cache derivatives.

```cpp
// Variables in AutoDiff

#include <AutoDiff/Core>     // for Variable and var
#include <AutoDiff/Basic>    // for Real

// creating variables
auto x = AutoDiff::var(2.0); // factory function
AutoDiff::Real y(2);         // constructor
auto z = y;                  // copy-construction

// accessing variables
auto value = x();            // get value
auto derivative = d(x);      // get derivative

// modifying variables
x = 3.0;                     // set value
x.setDerivative(1.0);        // set derivative
```

AutoDiff variables are instances of the `AutoDiff::Variable<Value, Derivative>` class template, which has two template parameters: the value type and the derivative type.
The next section explains how to choose suitable template parameters.

## Creating variables from values

### Using the factory

If you want to create a variable with a specific value, the `var` factory function is often the most convenient way to do so.
The correct variable type is automatically deduced by the compiler based on the function argument and the included [module](../index.md#modules) (e.g., `AutoDiff/Basic`).

```cpp
// Variable via factory function (Basic module)
#include <AutoDiff/Basic>      // module header
auto x = AutoDiff::var(2.0);   // Variable<double, double>; value = 2.0
auto y = AutoDiff::var(2);     // Variable<int, double>;    value = 2
auto z = AutoDiff::var<int>(); // Variable<int, double>;    value = 0
```

The same factory calls produce different variables when the Eigen module is included, because the default derivative type is different.

```cpp
// Variable via factory function (Eigen module)
#include <AutoDiff/Eigen>      // module header
#include <Eigen/Core>          // Eigen types
auto x = AutoDiff::var(2.0);   // Variable<double, Eigen::MatrixXd>; value = 2.0
auto y = AutoDiff::var(2);     // Variable<int, Eigen::MatrixXd>;    value = 2
auto z = AutoDiff::var<int>(); // Variable<int, Eigen::MatrixXd>;    value = 0
```

The function `var` is called a [factory](https://en.wikipedia.org/wiki/Factory_method_pattern), because it allows you to create variables without knowing the full type.
The full type is only available after certain type traits are specialized (usually in a module), which can occur later in the code.

### Using the constructor

If you want full control over the variable type, you can use the `Variable` class template directly.
The first template argument is the value type, and the second template argument is the derivative type.
Be aware that the combination of value and derivative types must be supported by the module you include.

```cpp
// Variable constructor (with template arguments)
AutoDiff::Variable<double, double> x(2); // value = 2.0
AutoDiff::Variable<float, double> y;     // value = 0.0f
```

For ease of use, modules offer predefined type aliases for frequently used combinations of value and derivative types.

```cpp
// Variable constructor (using alias)
#include <AutoDiff/Basic> // module defines aliases
AutoDiff::Real x(2);      // alias for Variable<double, double>; value = 2.0
AutoDiff::RealF y(2);     // alias for Variable<float, float>;   value = 2.0f
AutoDiff::Integer z;      // alias for Variable<int, double>;    value = 0
```

For more information about supported value types and the predefined type aliases, see the [module documentation](../index.md#modules).

## Variable getters and setters

At any time, you can access and modify the cached value and derivative of a variable.

### Getters

During an evaluation or differentiation pass, the value or derivative of a variable is cached and can be accessed through the following functions.

- **Call operator** `operator()`: returns the cached value of the variable.
- **Derivative operator** `d`: returns the cached derivative of the variable.
  This free function can be used without namespace because it is found through Argument-Dependent Lookup (ADL).

```cpp
// Variable getters
AutoDiff::Variable<Value, Derivative> x;
auto value = x();       // returns Value const&
auto derivative = d(x); // returns Derivative const&
```

> [!TIP]
> The syntax intentionally resembles mathematical function notation.
As described in [Mathematical notes on the API](../math/api-math.md), a variable `x` can be precisely considered a mathematical function $x$.
> Then, given a point $p$, the call `x()` returns $x(p)$ and `d(x)` returns the derivative $dx$, also written as $\mathrm{d}(x)$.

### Setters

- **Constructor**: the value is either passed as an argument or set to a default value.
  The derivative is always default-constructed.
- **Assignment operator**: lets you directly assign a new value to a variable for subsequent evaluations.
- **Derivative setter** `setDerivative`: sometimes it is necessary to set the derivative of *seed variables* before propagating the derivatives in a forward or reverse pass.
  For instance, use this function to set the vector (or covector) in a [Jacobian-vector product](../applications.md#jacobian-vector-products).

```cpp
// Variable setters
AutoDiff::Variable<Value, Derivative> x; // value and derivative default-constructed
x = value;                               // assign a new value
x.setDerivative(derivative);             // seed derivative propagation
```

## Copying variables

Variables are essentially smart pointers to a shared resource, and copying them is a lightweight operation.

```cpp
// Variable copy-construction
using AutoDiff::Real;
Real x;         // variable with default value 0.0
auto xCopy = x; // x and xCopy share same resource
xCopy = 2;      // set shared value
x();            // 2.0
```

Similar to a `std::shared_ptr`, assigning a variable `y` to another variable `x` copies the resource pointer of `y` to `x` and deallocates the previous resource of `x`, if no other variable points to it.

```cpp
// Variable assignment
using AutoDiff::Real;
Real x(2);
{
    Real y(3);
    x = y; // resource of x is deallocated and replaced with y's
    y = 4; // x and y now share the same resource
    x();   // 4.0
}          // end of scope deallocates y, but not its resource
x();       // 4.0, resource still owned by x
```

> [!IMPORTANT]
> Variables can only be copied or assigned if the value and derivative types match.
>
> ```cpp
> // Variable mismatch
> AutoDiff::Variable<double, double> x;
> AutoDiff::Variable<int,    double> y;
> AutoDiff::Variable<double, float>  z;
> y = x; // COMPILER ERROR: value type mismatch
> z = x; // COMPILER ERROR: derivative type mismatch
> ```

## Next steps

Next up: [Expressions](expression.md).

For a complete guide, see the [Documentation](../index.md).
