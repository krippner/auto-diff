// Copyright (c) 2025 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <algorithm> // for_each
#include <ranges>    // views::reverse
#include <sstream>
#include <stdexcept> // logic_error
#include <utility>   // move

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

Function::Function(AbstractVariable const& target)
{
    mSpecifiedTargets.obj.insert(target._node());
    setReferenceTarget();
}

Function::Function(Targets targets)
    : mSpecifiedTargets{std::move(targets)}
{
    setReferenceTarget();
}

Function::Function(Sources sources, Targets targets)
    : mSpecifiedSources{std::move(sources)}
    , mSpecifiedTargets{std::move(targets)}
{
    setReferenceTarget();
}

void Function::compile()
{
    mTargets.clear();
    mSources.clear();
    mPureTargets.clear();
    mPureSources.clear();
    mSequence.clear();
    try {
        std::ranges::for_each(TopoView(mSpecifiedTargets, mSpecifiedSources),
            [this](TopoView::NodeInfo const& current) {
                auto* computation = dynamic_cast<Computation*>(current.node);
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

auto Function::compiled() const -> bool { return !mSources.empty(); }

void Function::compileIfNecessary()
{
    if (!compiled()) {
        compile();
    }
}

auto Function::str() const -> std::string
{
    auto ss = std::ostringstream{};
    if (compiled()) {
        ss << "Function with " << mSources.size() << " sources, "
           << mTargets.size() << " targets, and " << mSequence.size()
           << " internal computations.\n";
        ss << "Sources:\n";
        std::ranges::for_each(mSources,
            [&](Computation* computation) { ss << computation << "\n"; });
        ss << "Targets:\n";
        std::ranges::for_each(mTargets,
            [&](Computation* computation) { ss << computation << "\n"; });
        ss << "Internal computations:\n";
        std::ranges::for_each(mSequence,
            [&](Computation* computation) { ss << computation << "\n"; });
    } else {
        ss << "Function not compiled.\n";
    }
    return ss.str();
}

void Function::evaluate()
{
    compileIfNecessary();
    std::ranges::for_each(
        mSequence, [](Computation* computation) { computation->evaluate(); });
    std::ranges::for_each(mPureTargets,
        [](Computation* computation) { computation->evaluate(); });
}

#ifndef AUTODIFF_NO_FORWARD_MODE
void Function::pushTangent()
{
    compileIfNecessary();
    std::ranges::for_each(mSequence,
        [](Computation* computation) { computation->pushTangent(); });
    std::ranges::for_each(mPureTargets,
        [](Computation* computation) { computation->pushTangent(); });
}

void Function::pushTangentAt(AbstractVariable const& seed)
{
    compileIfNecessary();

    auto* const seedNode = seed._node();

    if (mSources.find(seedNode) == mSources.end()) {
        throw SeedError("Seed variable must be a source of the function.");
    }

    auto const seedShape = seedNode->valueShape();
    std::ranges::for_each(mSources, [&](Computation* computation) {
        computation->setTangentZero(seedShape);
    });
    seedNode->setDerivativeIdentity();

    pushTangent();
}
#endif

#ifndef AUTODIFF_NO_REVERSE_MODE
void Function::pullGradient()
{
    compileIfNecessary();

    auto const seedShape = mReferenceTarget->derivativeCodomainShape();

    // initialize internal and source gradients to zero
    std::ranges::for_each(mSequence, [&](Computation* computation) {
        computation->setGradientZero(seedShape);
    });
    std::ranges::for_each(mPureSources, [&](Computation* computation) {
        computation->setGradientZero(seedShape);
    });

    // pull back gradients from targets to sources
    std::ranges::for_each(mPureTargets,
        [](Computation* computation) { computation->pullGradient(); });
    std::ranges::for_each(mSequence | std::views::reverse,
        [](Computation* computation) { computation->pullGradient(); });
}

void Function::pullGradientAt(AbstractVariable const& seed)
{
    compileIfNecessary();

    auto* const seedNode = seed._node();

    if (mTargets.find(seedNode) == mTargets.end()) {
        throw SeedError("Seed variable must be a target of the function.");
    }

    auto const seedShape = seedNode->valueShape();
    std::ranges::for_each(mTargets, [&](Computation* computation) {
        computation->setGradientZero(seedShape);
    });
    seedNode->setDerivativeIdentity();

    pullGradient();
}
#endif

void Function::setReferenceTarget()
{
    if (mSpecifiedTargets.obj.empty()) {
        throw EmptyFunctionError("Function must have at least one target.");
    }
    mReferenceTarget
        = dynamic_cast<Computation*>(*mSpecifiedTargets.obj.begin());
}

template <typename... Variables>
auto from(Variables const&... variables) -> Function::Sources
{
    Function::Sources sources{};
    (sources.obj.insert(variables._node()), ...);
    return sources;
}

template <typename... Variables>
auto to(Variables const&... variables) -> Function::Targets
{
    Function::Targets targets{};
    (targets.obj.insert(variables._node()), ...);
    return targets;
}

} // namespace AutoDiff
