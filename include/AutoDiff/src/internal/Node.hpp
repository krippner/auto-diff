// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_INTERNAL_NODE_HPP
#define AUTODIFF_SRC_INTERNAL_NODE_HPP

#include "range_algorithm.hpp"

#include <memory>
#include <queue>
#include <unordered_set>
#include <utility> // move
#include <vector>

namespace AutoDiff::internal {

/**
 * @brief Used to preview ownership of a node.
 *
 * For details see @c Node::addParentOwner and @c Node::removeParentOwner.
 */
struct NodeOwner { };

/**
 * @class Node
 * @brief Graph node with owning pointers to its children.
 *
 * On destruction, child nodes are guaranteed to be destroyed iteratively.
 * This prevents stack-overflow when destructing large graphs.
 */
class Node {
public:
    using NodePtr  = std::shared_ptr<Node>;
    using PtrSet   = std::unordered_set<NodePtr>;
    using OwnerPtr = std::unique_ptr<NodeOwner>;

    // Nodes are unique
    Node(Node const&)                    = delete;
    auto operator=(Node const&) -> Node& = delete;

    /**
     * @brief Destruct this node and its exclusively owned successors.
     *
     * Successors are not destructed if they are owned (via @c addParentOwner())
     * outside the subgraph of this node.
     * Iterative node destruction prevents stack-overflow which could occur when
     * calling the destructors recursively.
     */
    virtual ~Node()
    {
        if (!mChildren.empty()) {
            deleteIteratively(this);
        } // otherwise, no iteration necessary
    }

    /**
     * @brief The children nodes of this node.
     */
    [[nodiscard]] auto children() const -> PtrSet const& { return mChildren; }

    /**
     * @brief Add child node as shared resource, if not already present.
     *
     * Ownership is released by @c releaseChildren() or destructor.
     *
     * @param child    the node to be inserted
     */
    void addChild(NodePtr child)
    {
        auto const [iter, inserted] = mChildren.emplace(std::move(child));
        if (inserted) {
            (*iter)->addParentOwner(mOwner);
        }
    }

    /**
     * @brief Register ownership for this node.
     *
     * Must be called when creating a shared_ptr to this node
     * to ensure all its successors will be kept alive.
     *
     * @param  owner       the unique pointer to the owner
     */
    void addParentOwner(OwnerPtr const& owner)
    {
        mParentOwners.insert(owner.get());
    }

    /**
     * @brief Unregister ownership for this node.
     *
     * Must be called when resetting/deleting a shared_ptr to this node
     * to ensure all its successors are destructed iteratively.

     * @param  owner       the unique pointer to the owner
     */
    void removeParentOwner(OwnerPtr const& owner)
    {
        mParentOwners.erase(owner.get());
    }

    /**
     * @brief Release ownership of the children nodes.
     */
    void releaseChildren()
    {
        if (mChildren.empty()) {
            return;
        }
        removeOwnership();
        removeChildren();
    }

protected:
    Node()                                   = default;
    Node(Node&&) noexcept                    = default;
    auto operator=(Node&&) noexcept -> Node& = default;

private:
    static void deleteIteratively(Node* root)
    {
        std::vector<Node*> mNodesToDelete{};
        auto queue = std::queue<Node*>();
        queue.push(root);
        // breadth-first search for nodes that can be deleted
        while (!queue.empty()) {
            Node* node = queue.front();
            node->removeOwnership();
            mNodesToDelete.push_back(node);
            for_each_in_range(
                queue.front()->mChildren, [&](NodePtr const& child) {
                    if (child->canBeDeleted()) {
                        queue.push(child.get());
                    }
                });
            queue.pop();
        }
        for_each_in_reversed_range(
            mNodesToDelete, [](Node* node) { node->removeChildren(); });
    }

    void removeOwnership()
    {
        for_each_in_range(mChildren, [this](NodePtr const& child) {
            child->removeParentOwner(this->mOwner);
        });
    }

    [[nodiscard]] auto canBeDeleted() const -> bool
    {
        return mParentOwners.empty(); // preview shows no owners
    }

    void removeChildren() { mChildren.clear(); }

    PtrSet mChildren; // unique edges

    // to register ownership with children
    OwnerPtr mOwner{std::make_unique<NodeOwner>()};
    using OwnerSet = std::unordered_set<NodeOwner*>;
    OwnerSet mParentOwners; // preview ownership
};

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_NODE_HPP
