#include <AutoDiff/Basic>
#include <AutoDiff/Core>

#include <cstdio>

int main()
{
    // Function definition:
    std::printf("f: R ⨉ R → R, (x, y) ↦ z = x y\n");
    AutoDiff::Real x, y, z;
    AutoDiff::Function f(from(x, y), to(z));
    z = x * y;

    // Evaluation:
    x = 2, y = 3, f.evaluate(); // z = f(2, 3)
    std::printf("f(2, 3) = %.f\n", z());

    // Gradient computation (reverse-mode AD):
    f.pullGradientAt(z); // <∂z/∂x ∂z/∂y> at (2, 3)
    std::printf("grad f  = <%.f %.f>\n", d(x), d(y));

    // Tangent vectors (forward-mode AD):
    f.pushTangentAt(x); // tangent along x at (2, 3)
    std::printf("∂f/∂x   = %.f\n", d(z));
    f.pushTangentAt(y); // tangent along y at (2, 3)
    std::printf("∂f/∂y   = %.f\n", d(z));
}
