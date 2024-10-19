# AutoDiff: automatic differentiation framework for C++

[![CI Builds](https://github.com/krippner/auto-diff/actions/workflows/ci.yml/badge.svg)](https://github.com/krippner/auto-diff/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/krippner/auto-diff/badge.svg?branch=main)](https://coveralls.io/github/krippner/auto-diff?branch=main)

Welcome to AutoDiff, a modern C++17 header-only library for **automatic differentiation (AD)** in forward- and reverse mode.
Unlike other AD libraries, AutoDiff is a **framework** that provides the generic building blocks for AD, allowing you to choose what data types to compute with and to create custom AD implementations with ease.

AD is an **efficient algorithm** for computing **exact derivatives** of (usually numeric) functions.
It is a standard tool in numerous fields, including optimization, machine learning, and scientific computing.

Python bindings for this library are provided in a separate repository.
For more information, see the [Python bindings repository](https://github.com/krippner/auto-diff-python).

## Features

- **Modular design** [based on differential geometry](docs/math/diff-geo.md#top)
  - [*Core* differentiation framework](docs/index.md#core-framework): smooth functions, pushforward and pullback
  - [*Basic* module](docs/modules/basic.md#top): lightweight implementation with C++ built-in types for scalar computations
  - [*Eigen* module](docs/modules/eigen.md#top): fast implementation using [Eigen](https://en.wikipedia.org/wiki/Eigen_(C%2B%2B_library)) (requires Eigen 3.4)
    - Array computations
    - Linear algebra with vectors and dense matrices
    - Coefficient-wise operations with NumPy-style broadcasting
  - Easily [extend or add modules](docs/developer.md#top) for custom types: e.g. for symbolic computation
- **Dynamic computation**: what you compute is what you differentiate
  - Regular C++ [control flow](docs/applications.md#control-flow): function calls, loops, branches
  - (Optional) [eager evaluation](docs/core/expression.md#eager-evaluation): debug and inspect intermediate results
  - [Lazy evaluation](docs/core/function.md#evaluating-functions): efficient re-evaluations, offering precise control over what to evaluate
- **High-performance computation** with compile-time expression optimization
  - Faster algorithms and fewer temporaries thanks to Eigen's [expression templates](https://en.wikipedia.org/wiki/Expression_templates)
  - Memory efficient (see [Expressions vs. variables](docs/core/expression.md#expressions-vs-variables))
  - Intuitive mathematical syntax with operator overloading
- **Automatic memory management** using [RAII](docs/core/expression.md#memory-management-in-expressions)
  - Resources in dynamic computations are released iteratively, preventing stack overflow
  - No hidden side effects: no global state as in tape-based auto-diff implementations
- **Lightweight** library
  - Header-only: just copy the `include` directory to your project
  - Separate module headers: include only what you need
  - Integration is straightforward with [CMake presets](#using-it-with-cmake) and [Conan integration](#using-it-with-conan)
- **Clean and flexible** code base
  - Warning-free even on high warning levels such as `-Wall`, `-Wextra`, `-pedantic`
  - Follows [SOLID](https://en.wikipedia.org/wiki/SOLID) design principles
  - Extensive [unit tests](#building-the-unit-tests)
  - Linted with [Clang-Tidy](https://clang.llvm.org/extra/clang-tidy/)

For a complete guide, please see the [documentation](docs/index.md#top).

## Examples

### Eigen example

This example computes the (element-wise) gradient of the product of two arrays.
Values and derivatives have `Eigen::Array` type.

```cpp
// Example: gradient computation with Eigen arrays
#include <AutoDiff/Core>   // for Function, var, and d
#include <AutoDiff/Eigen>  // support for Eigen types
#include <Eigen/Core>      // for Eigen::Array
#include <iostream>

using std::cout, AutoDiff::var, AutoDiff::Function;

int main() {
  // Create two 1D array variables (transposed for horizontal output)
  auto x = var(Eigen::Array3d{1, 2, 3}.transpose());
  auto y = var(Eigen::Array3d{4, 5, 6}.transpose());

  // Assign their (element-wise) product to a new variable
  auto z = var(x * y);

  // Variables are evaluated eagerly
  cout << "z = " << z() << '\n';      // z = 4 10 18

  // Create the function f : (x, y) ↦ z = x * y
  Function f(z); // short for: Function f(from(x, y), to(z))

  // Compute the gradient of f at (x, y) using reverse-mode AD
  f.pullGradientAt(z);

  // Get the components of the (element-wise) gradient
  cout << "∇_x f = " << d(x) << '\n'; // ∇_x f = 4 5 6
  cout << "∇_y f = " << d(y) << '\n'; // ∇_y f = 1 2 3
}
```

### Floating-point example

The following example computes the gradient of a scalar function $f(x, y) = xy$.
Values and derivatives have type `double`.
Note that the value of $z$ is computed lazily, i.e., not until `f.evaluate()` is called.

```cpp
// Example: gradient computation with double variables
#include <AutoDiff/Core>   // for Function and d
#include <AutoDiff/Basic>  // for Real
#include <iostream>

using std::cout, AutoDiff::Real, AutoDiff::Function;

int main() {
  // Create the function f : R ⨉ R → R, (x, y) ↦ z = x * y
  Real x, y, z; // floating-point variables
  Function f(from(x, y), to(z));
  z = x * y;

  // Lazy evaluation
  x = 2, y = 3;
  f.evaluate();
  cout << "f(2, 3) = " << z() << '\n'; // f(2, 3) = 6

  // Compute the gradient of f at (x, y) using reverse-mode AD
  f.pullGradientAt(z);

  // Get the components of the gradient
  cout << "∂f/∂x = " << d(x) << '\n'; // ∂f/∂x = 3
  cout << "∂f/∂y = " << d(y) << '\n'; // ∂f/∂y = 2
}
```

### Training a neural network with backpropagation

See [examples/backpropagation.cpp](examples/backpropagation.cpp) for the documented source code.
Steps to compile and run the example:

1. Download the repository

    ```bash
    git clone https://github.com/krippner/auto-diff.git
    cd auto-diff   
    ```

2. If you don't have Eigen installed, you can install it on Ubuntu with:

    ```bash
    sudo apt install libeigen3-dev
    ```

3. To compile this example with GCC/Clang, run (in the AutoDiff root directory):

    ```bash
    cmake --preset examples-ninja-linux-release
    cmake --build build/Release --target Backpropagation
    ```

## Using it with CMake

### Including it in your project with CMake

If you are using CMake, you can use the `FetchContent` module to download and include the library in your project.
Here is an example of how to use AutoDiff in a CMake project.

Create a `CMakeLists.txt` file:

```cmake
cmake_minimum_required(VERSION 3.16)

project(Example)

include(FetchContent)
FetchContent_Declare(
  autodiff
  GIT_REPOSITORY https://github.com/krippner/auto-diff.git
  GIT_TAG v0.4.0
)
FetchContent_MakeAvailable(autodiff)

add_executable(Example main.cpp)
target_link_libraries(Example PRIVATE AutoDiff::AutoDiff)
```

Create a `main.cpp` file:

```cpp
#include <AutoDiff/Core>
#include <AutoDiff/Basic>
#include <iostream>
int main()
{
    auto x = AutoDiff::var(1.0);
    auto y = AutoDiff::var(log(x));
    std::cout << "y(1.0) = " << y() << '\n';
    AutoDiff::Function f(y);
    f.pushTangentAt(x);
    std::cout << "df/dx = " << d(y) << '\n';
}
```

Build the project:

```bash
mkdir build && cd build
cmake .. -DFETCHCONTENT_UPDATES_DISCONNECTED=ON
cmake --build .
```

This will download the repository into your build folder and build the project.
Use the setting `FETCHCONTENT_UPDATES_DISCONNECTED=ON` to avoid downloading the repository every time you configure your project.

### Building the unit tests

Make sure you have `Catch2` (v3) and `Eigen` (v3.4) installed.

Download the repository and configure CMake/Clang:

```bash
git clone https://github.com/krippner/auto-diff.git
cd auto-diff
cmake --preset tests-ninja-linux-release
```

For Visual Studio, you can use the `tests-msvc2022` preset.

Build and run:

```bash
cd build/Release
cmake --build .
ctest
```

## Using it with Conan

### Installation with Conan

Download the repository and install the library to your Conan cache:

```bash
git clone https://github.com/krippner/auto-diff.git
cd auto-diff
conan create . --build=missing
```

This will also build and run the unit tests, unless your Conan configuration `tools.build:skip_test` is set to `True`.

### Including it in your project with Conan

Add the following line to your `conanfile.txt`:

```ini
[requires]
autodiff/0.4.0
```

To make the library available in your CMake project, run:

```bash
conan install . --build=missing
```

This will also print instructions on how to include the library in your CMake project.
