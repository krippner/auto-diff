#ifndef AUTODIFF_SRC_INTERNAL_REFERENCE_H
#define AUTODIFF_SRC_INTERNAL_REFERENCE_H

#include "Computation.hpp"
#include "Node.hpp" // NodeOwner

#include <algorithm> // swap
#include <memory>

namespace AutoDiff::internal {

/**
 * @class Reference
 * @brief Essentially a shared pointer to a computation node.
 *
 * Additionally, it holds a unique owner object that is used
 * to register and unregister ownership of the computation.
 */
template <typename Value, typename Derivative>
class Reference {
public:
    using Owner       = internal::NodeOwner;
    using Computation = internal::Computation<Value, Derivative>;

    Reference() { mComputation->addParentOwner(mOwner); }

    Reference(Reference const& other)
        : mComputation{other.mComputation}
        , mComputationPtr{other.mComputationPtr}
    {
        mComputation->addParentOwner(mOwner); // owner is unique
    }

    auto operator=(Reference other) -> Reference&
    {
        swap(*this, other);
        return *this;
    }

    ~Reference()
    {
        if (static_cast<bool>(mComputation)) {
            mComputation->removeParentOwner(mOwner);
        } // else transferOperationTo was called
    }

    Reference(Reference&&) noexcept                    = default;
    auto operator=(Reference&&) noexcept -> Reference& = default;

    [[nodiscard]] auto operator->() const -> Computation*
    {
        // Note: raw ptr always valid:
        // ~Node guarantees that this function is not called
        // between ~Computation and ~Variable.
        return mComputationPtr;
    }

    void transferOperationTo(internal::Node& node)
    {
        // Note: shared_ptr always valid:
        // This function is called at most once, which is when the
        // parent Variable is bound in an expression (copy ctor).

        // unregister ownership
        mComputation->removeParentOwner(mOwner);
        // transfer owning pointer to node
        node.addChild(mComputation);
        mComputation.reset();
    }

    [[nodiscard]] friend auto operator==(
        Reference const& left, Reference const& right)
    {
        return left.mComputationPtr == right.mComputationPtr;
    }

    [[nodiscard]] friend auto operator!=(
        Reference const& left, Reference const& right)
    {
        return left.mComputationPtr != right.mComputationPtr;
    }

    friend void swap(Reference& a, Reference& b) noexcept
    {
        using std::swap;
        swap(a.mOwner, b.mOwner);
        swap(a.mComputation, b.mComputation);
        swap(a.mComputationPtr, b.mComputationPtr);
    }

private:
    std::unique_ptr<Owner> mOwner{std::make_unique<Owner>()};
    std::shared_ptr<Computation> mComputation{std::make_shared<Computation>()};
    Computation* mComputationPtr{mComputation.get()};
};

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_REFERENCE_H
