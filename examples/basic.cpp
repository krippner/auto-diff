// Example: gradient computation with double variables
#include <AutoDiff/Basic>
#include <AutoDiff/Core>
#include <iostream>

using std::cout, AutoDiff::Real, AutoDiff::Function;

int main()
{
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
