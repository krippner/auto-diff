// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_CORE_FUNCTION_HPP
#define AUTODIFF_SRC_CORE_FUNCTION_HPP

#include "../internal/AbstractComputation.hpp"
#include "../internal/TopoView.hpp"
#include "../internal/range_algorithm.hpp"
#include "AbstractVariable.hpp"

#include <sstream>
#include <stdexcept> // logic_error
#include <unordered_set>
#include <utility> // move
#include <vector>

namespace AutoDiff {

/**
 * @class EmptyFunctionError
 * @brief A function must have at least one target.
 */
class EmptyFunctionError : public std::logic_error {
public:
    explicit EmptyFunctionError(std::string const& arg)
        : logic_error(arg)
    {
    }
};

/**
 * @class CyclicDependencyError
 * @brief Expressions with cyclic dependencies cannot be evaluated.
 *
 * Cyclic dependencies between variables can be introduced when
 * assigning certain expressions to variables.
 */
class CyclicDependencyError : public std::logic_error {
public:
    explicit CyclicDependencyError(std::string const& arg)
        : logic_error(arg)
    {
    }
};

/**
 * @class SeedError
 * @brief Derivative propagation fails if the wrong variable is seeded.
 */
class SeedError : public std::logic_error {
public:
    explicit SeedError(std::string const& arg)
        : logic_error(arg)
    {
    }
};

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
    explicit Function(AbstractVariable const& target)
    {
        mSpecifiedTargets.obj.insert(target._node());
        setReferenceTarget();
    }

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
    explicit Function(Targets targets)
        : mSpecifiedTargets{std::move(targets)}
    {
        setReferenceTarget();
    }

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
    Function(Sources sources, Targets targets)
        : mSpecifiedSources{std::move(sources)}
        , mSpecifiedTargets{std::move(targets)}
    {
        setReferenceTarget();
    }

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
    void compile()
    {
        mTargets.clear();
        mSources.clear();
        mPureTargets.clear();
        mPureSources.clear();
        mSequence.clear();
        try {
            internal::for_each_in_range(
                TopoView(mSpecifiedTargets, mSpecifiedSources),
                [this](TopoView::NodeInfo const& current) {
                    auto* computation
                        = dynamic_cast<Computation*>(current.node);
                    // being source and being target are independent properties,
                    // need to consider all 4 cases
                    if (current.isLeaf) {
                        this->mSources.insert(computation);
                        if (current.isRoot) {
                            this->mTargets.insert(computation);
                        } else {
                            // source but not target
                            this->mPureSources.insert(computation);
                        }
                    } else if (current.isRoot) {
                        this->mTargets.insert(computation);
                        // target but not source
                        this->mPureTargets.insert(computation);
                    } else {
                        // Collect internal computations in a topologically
                        // ordered sequence.
                        this->mSequence.push_back(computation);
                    }
                });
        } catch (internal::CyclicGraphError const& /*error*/) {
            // prevent evaluation
            mTargets.clear();
            mSequence.clear();
            mPureTargets.clear();
            mPureSources.clear();

            // indicate failed compilation
            mSources.clear();

            throw CyclicDependencyError(
                "Cyclic dependency detected during function compilation.");
        }
    }

    /**
     * @brief True if the function has been compiled successfully.
     */
    auto compiled() const -> bool { return !mSources.empty(); }

    /**
     * @brief Compile the function if it is not already successfully compiled.
     */
    void compileIfNecessary()
    {
        if (!compiled()) {
            compile();
        }
    }

    /**
     * @brief Info string about the function's internals.
     *
     * This function is intended for debugging purposes.
     */
    auto str() const -> std::string
    {
        auto ss = std::ostringstream{};
        if (compiled()) {
            ss << "Function with " << mSources.size() << " sources, "
               << mTargets.size() << " targets, and " << mSequence.size()
               << " internal computations.\n";
            ss << "Sources:\n";
            internal::for_each_in_range(mSources,
                [&](Computation* computation) { ss << computation << "\n"; });
            ss << "Targets:\n";
            internal::for_each_in_range(mTargets,
                [&](Computation* computation) { ss << computation << "\n"; });
            ss << "Internal computations:\n";
            internal::for_each_in_range(mSequence,
                [&](Computation* computation) { ss << computation << "\n"; });
        } else {
            ss << "Function not compiled.\n";
        }
        return ss.str();
    }

    /**
     * @brief Evaluate the target and intermediate variables.
     *
     * Before the first evaluation, the function is automatically compiled
     * if necessary.

     * @note Before calling this, all source variables must have valid values.
     */
    void evaluate()
    {
        compileIfNecessary();
        internal::for_each_in_range(mSequence,
            [](Computation* computation) { computation->evaluate(); });
        internal::for_each_in_range(mPureTargets,
            [](Computation* computation) { computation->evaluate(); });
    }

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
    void pushTangent()
    {
        compileIfNecessary();
        internal::for_each_in_range(mSequence,
            [](Computation* computation) { computation->pushTangent(); });
        internal::for_each_in_range(mPureTargets,
            [](Computation* computation) { computation->pushTangent(); });
    }

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
    void pushTangentAt(AbstractVariable const& seed)
    {
        compileIfNecessary();

        auto* const seedNode = seed._node();

        if (mSources.find(seedNode) == mSources.end()) {
            throw SeedError("Seed variable must be a source of the function.");
        }

        auto const seedShape = seedNode->valueShape();
        internal::for_each_in_range(mSources, [&](Computation* computation) {
            computation->setTangentZero(seedShape);
        });
        seedNode->setDerivativeIdentity();

        pushTangent();
    }
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
    void pullGradient()
    {
        compileIfNecessary();

        auto const seedShape = mReferenceTarget->derivativeCodomainShape();

        // initialize internal and source gradients to zero
        internal::for_each_in_range(mSequence, [&](Computation* computation) {
            computation->setGradientZero(seedShape);
        });
        internal::for_each_in_range(
            mPureSources, [&](Computation* computation) {
                computation->setGradientZero(seedShape);
            });

        // pull back gradients from targets to sources
        internal::for_each_in_range(mPureTargets,
            [](Computation* computation) { computation->pullGradient(); });
        internal::for_each_in_reversed_range(mSequence,
            [](Computation* computation) { computation->pullGradient(); });
    }

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
    void pullGradientAt(AbstractVariable const& seed)
    {
        compileIfNecessary();

        auto* const seedNode = seed._node();

        if (mTargets.find(seedNode) == mTargets.end()) {
            throw SeedError("Seed variable must be a target of the function.");
        }

        auto const seedShape = seedNode->valueShape();
        internal::for_each_in_range(mTargets, [&](Computation* computation) {
            computation->setGradientZero(seedShape);
        });
        seedNode->setDerivativeIdentity();

        pullGradient();
    }
#endif

private:
    using Computation = internal::AbstractComputation;
    using TopoView    = internal::TopoView;

    void setReferenceTarget()
    {
        if (mSpecifiedTargets.obj.empty()) {
            throw EmptyFunctionError("Function must have at least one target.");
        }
        mReferenceTarget
            = dynamic_cast<Computation*>(*mSpecifiedTargets.obj.begin());
    }

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
auto from(Variables const&... variables) -> Function::Sources
{
    Function::Sources sources{};
    (sources.obj.insert(variables._node()), ...);
    return sources;
}

/**
 * @brief Create a set of function targets from a list of variables.
 *
 * @param  variables   the target variables
 */
template <typename... Variables>
auto to(Variables const&... variables) -> Function::Targets
{
    Function::Targets targets{};
    (targets.obj.insert(variables._node()), ...);
    return targets;
}

} // namespace AutoDiff

#endif // AUTODIFF_SRC_CORE_FUNCTION_HPP
