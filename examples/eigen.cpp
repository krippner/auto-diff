#include <AutoDiff/Core>
#include <AutoDiff/Eigen>
#include <Eigen/Core>
#include <iostream>

using AutoDiff::var, AutoDiff::Real, AutoDiff::Function, AutoDiff::Expression;
using std::cout;

// This function can take any AutoDiff expression as input
// and returns the sigmoid function applied element-wise.
template <typename X>
auto logistic(Expression<X> const& x)
{
    return 1 / (1 + exp(-4 * x)); // scalars broadcast to arrays
}

// Branches can only be differentiated through special functions
template <typename X>
auto reLU(Expression<X> const& x)
{
    return max(x);
}

int main()
{
    {
        cout << "\nGradient computation with Eigen arrays\n";
        // Create two 1D array variables (transposed for nicer output)
        auto x = var(Eigen::Array3d{1, 2, 3}.transpose());
        auto y = var(Eigen::Array3d{4, 5, 6}.transpose());

        // Assign their (element-wise) product to a new variable
        auto z = var(x * y);

        // Variables are evaluated eagerly
        cout << "z = " << z() << '\n'; // z = 4 10 18

        // Create the function f : (x, y) ↦ z = x * y
        Function f(z); // short for: Function f(from(x, y), to(z))

        // Compute the gradient of f at (x, y) using reverse-mode AD
        f.pullGradientAt(z);

        // Get the components of the (element-wise) gradient
        cout << "∇_x f = " << d(x) << '\n'; // ∇_x f = 4 5 6
        cout << "∇_y f = " << d(y) << '\n'; // ∇_y f = 1 2 3
    }
    {
        cout << "\nPassing expressions to functions\n";
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
    }
    {
        cout << "\nLoop example\n";
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
    }
    {
        cout << "\nConditional expressions\n";
        auto x = var(Eigen::Array3d{-1, 0, 1}.transpose());
        auto y = var(reLU(x));
        cout << "y = " << y() << '\n'; // y = 0 0 1
    }
    {
        cout << "\nCaution: if statements cannot depend on variables\n";
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
    }
    {
        cout << "\nComputing the Jacobian matrix\n";
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
    }
    {
        cout << "\nGradient computation\n";
        auto x = var(Eigen::Vector3d{1, 2, 3});
        auto y = var(norm(x)); // L²-norm

        Function f(y); // f : R³ → R, x ↦ y = ||x||
        f.pullGradientAt(y);
        cout << "∇f = " << d(x) << '\n'; // ∇f = 0.267261 0.534522 0.801784
    }
    {
        cout << "\nElement-wise gradient computation\n";
        auto x = var(Eigen::Vector3d{1, 2, 3});
        auto y = var(Eigen::Vector3d{4, 5, 6});
        auto z = var(cwiseProduct(x, y));

        Function f(z);
        z.setDerivative(Eigen::RowVector3d::Ones());
        f.pullGradient();
        cout << "∇_x f = " << d(x) << '\n'; // ∇_x f = 4 5 6
        cout << "∇_y f = " << d(y) << '\n'; // ∇_y f = 1 2 3
    }
    {
        cout << "\nDirectional derivative (Jacobian-vector product)\n";
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
    }
}
