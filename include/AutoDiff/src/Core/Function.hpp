// Copyright (c) 2024-2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_CORE_FUNCTION_HPP
#define AUTODIFF_SRC_CORE_FUNCTION_HPP

#include "../internal/AbstractComputation.hpp"
#include "../internal/TopoView.hpp"
#include "AbstractVariable.hpp"

#include <unordered_set>
#include <vector>

namespace AutoDiff {

/**
 * @class Function
 * @brief Represents a program defined by target variables as functions of
 * source variables for evaluation and differentiation.
 *
 * In maths, the space containing sources or targets is usually called the
 * function domain or codomain, respectively.
 *
 * Note 1:
 * Generally, the function needs to be evaluated before differentiating,
 * either lazily during expression construction or explicitly by calling
 * @c evaluate.
 *
 * Note 2:
 * After assigning a new expression to one of the variables involved,
 * the function must be re-compiled by calling @c compile explicitly.
 *
 * This is necessary because a @c Function object is just a view into the
 * internal computation graph and it holds only non-owning references
 * to the computation nodes (which are owned by variables).
 */
class Function {
public:
    // translate graph terminology

    // Set of function targets
    using Targets = internal::TopoView::Roots;
    // Set of function sources
    using Sources = internal::TopoView::Leaves;

    /**
     * @brief Create a function with a single target variable.
     *
     * The source variables are determined automatically by traversing the
     * computation graph starting from the target variable.
     *
     * @param  target      the target variable
     */
    inline explicit Function(AbstractVariable const& target);

    /**
     * @brief Create a function with multiple target variables.
     *
     * Use the @c to function to create a @c Targets object.
     *
     * The source variables are determined automatically by traversing the
     * computation graph starting from the target variables.
     *
     * @param  targets     the target variables, must not be empty
     *
     * @throws EmptyFunctionError, if the function has no target.
     */
    inline explicit Function(Targets targets);

    /**
     * @brief Create a function mapping sources to targets.
     *
     * The source variables are used to limit the search for dependencies.
     * This can be useful to partition the computation graph into subgraphs.
     *
     * @code{.cpp}
     * auto x = var(..);
     * auto [u, v] = expression_1(x);
     * auto [a, b] = expression_2(u, v);
     * auto f_1_2 = Function(from(x), to(a, b));    // x ↦ (a, b)
     * auto f_2   = Function(from(u, v), to(a, b)); // (u, v) ↦ (a, b)
     * @endcode
     *
     *
     * @param  sources     the source variables;
     *                     need not be actual sources of the function
     * @param  targets     the target variables, must not be empty
     *
     * @throws EmptyFunctionError, if the function has no target.
     */
    inline Function(Sources sources, Targets targets);

    ~Function() = default;

    Function(Function const&)                        = default;
    Function(Function&&) noexcept                    = default;
    auto operator=(Function const&) -> Function&     = default;
    auto operator=(Function&&) noexcept -> Function& = default;

    /**
     * @brief Compile the function for evaluation and differentiation.
     *
     * Compilation generates a topologically ordered sequence of computation
     * references, which is used to efficiently traverse the computation graph.
     * It is triggered automatically before the first evaluation or
     * differentiation.
     *
     * @note This function must be called after assigning a new expression
     * to one of the variables involved.
     *
     * @throws CyclicDependencyError, if the program has cyclic dependencies.
     */
    inline void compile();

    /**
     * @brief True if the function has been compiled successfully.
     */
    inline auto compiled() const -> bool;

    /**
     * @brief Compile the function if it is not already successfully compiled.
     */
    inline void compileIfNecessary();

    /**
     * @brief Info string about the function's internals.
     *
     * This function is intended for debugging purposes.
     */
    inline auto str() const -> std::string;

    /**
     * @brief Evaluate the target and intermediate variables.
     *
     * Before the first evaluation, the function is automatically compiled
     * if necessary.

     * @note Before calling this, all source variables must have valid values.
     */
    inline void evaluate();

#ifndef AUTODIFF_NO_FORWARD_MODE
    /**
     * @brief Forward-mode automatic differentiation.
     *
     * Computes the tangent vectors at target and intermediate variables
     * by propagating the derivatives related to the source variables forward
     * along the function, i.e., in the same direction as the evaluation.
     *
     * Use this member function to compute the Jacobian-vector product.
     *
     * @code{.cpp}
     * auto x = var(0);          // literal variable
     * auto u = var(x * 2)       // eagerly evaluated variable
     * auto f = Function(u);
     * auto delta_x = 1.0;       // (scalar) tangent vector
     * x.setDerivative(delta_x); // seed forward propagation
     * f.pushTangent();          // compute the Jacobian-vector product
     * d(u);                     // δu = ∂u/∂x * δx = 2.0
     * @endcode
     *
     * @note Before calling this, the function must be evaluated and all source
     * variables must have valid derivatives.
     */
    inline void pushTangent();

    /**
     * @brief Forward-mode automatic differentiation with seed.
     *
     * Differentiates the target and intermediate variables of the function
     * with respect to a specified source variable (seed).
     *
     * Use this member function to compute the Jacobian matrix.
     *
     * @param  seed        the source variable used to seed propagation
     *
     * @code{.cpp}
     * auto x = var(0);       // literal variable
     * auto u = var(x * 2)    // eagerly evaluated variable
     * auto f = Function(u);
     * f.pushTangentAt(x);    // compute the Jacobian matrix
     * d(u);                  // ∂u/∂x = 2.0
     * @endcode
     *
     * @note Before calling this, the function must be evaluated.
     *
     * @throws SeedError, if @c seed is not an actual source of the function.
     */
    inline void pushTangentAt(AbstractVariable const& seed);
#endif

#ifndef AUTODIFF_NO_REVERSE_MODE
    /**
     * @brief Reverse-mode automatic differentiation (backpropagation).
     *
     * Computes the gradients with respect to source and intermediate variables
     * by propagating the derivatives related to the target variables backward
     * along this function, i.e., in the opposite direction of the evaluation.
     *
     * @code{.cpp}
     * auto x = var(0);          // literal variable
     * auto u = var(x * 2)       // eagerly evaluated variable
     * auto f = Function(u);
     * auto nabla_u = 1.0;       // (scalar) gradient w.r.t. u
     * u.setDerivative(nabla_u); // seed backpropagation
     * f.pullGradient();
     * d(x);                     // ∇_x = ∇_u * ∂u/∂x = 2.0
     * @endcode
     *
     * @note Before calling this, the function must be evaluated and all target
     * variables must have valid derivatives.
     */
    inline void pullGradient();

    /**
     * @brief Reverse-mode automatic differentiation (backpropagation) with
     * seed.
     *
     * Differentiates the specified target variable (seed) with respect
     * to the source and intermediate variables of the function.
     *
     * Use this method to compute the gradient.
     *
     * @param  seed        the target variable used to seed backpropagation
     *
     * @code{.cpp}
     * auto x = var(0);      // literal variable
     * auto u = var(x * 2)   // eagerly evaluated variable
     * auto f = Function(u);
     * f.pullGradientAt(u);  // compute the gradient (or Jacobian matrix)
     * d(x);                 // ∇_x = ∂u/∂x = 2.0
     * @endcode
     *
     * @note Before calling this, the function must be evaluated.
     *
     * @throws SeedError, if @c seed is not a target of the function.
     */
    inline void pullGradientAt(AbstractVariable const& seed);
#endif

private:
    using Computation = internal::AbstractComputation;
    using TopoView    = internal::TopoView;

    inline void setReferenceTarget();

    // user specified
    Sources mSpecifiedSources{};
    Targets mSpecifiedTargets{};
    Computation* mReferenceTarget{nullptr}; // target that always exists

    // populated during compilation
    std::unordered_set<Computation*> mSources;
    std::unordered_set<Computation*> mTargets;
    std::unordered_set<Computation*> mPureSources; // not targets
    std::unordered_set<Computation*> mPureTargets; // not sources
    std::vector<Computation*> mSequence;           // internal nodes
};

/**
 * @brief Create a set of function sources from a list of variables.
 *
 * @param  variables   the source variables
 */
template <typename... Variables>
auto from(Variables const&... variables) -> Function::Sources;

/**
 * @brief Create a set of function targets from a list of variables.
 *
 * @param  variables   the target variables
 */
template <typename... Variables>
auto to(Variables const&... variables) -> Function::Targets;

} // namespace AutoDiff

#include "Function.tpp" // implementation

#endif // AUTODIFF_SRC_CORE_FUNCTION_HPP
