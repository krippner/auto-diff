// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_INTERNAL_TOPO_VIEW_HPP
#define AUTODIFF_SRC_INTERNAL_TOPO_VIEW_HPP

#include "Node.hpp"
#include "Path.hpp"
#include "range_algorithm.hpp"

#include <iterator>
#include <unordered_set>
#include <utility> // move

namespace AutoDiff::internal {

/**
 * @class TopoView
 * @brief Topologically ordered view on a directed acyclic graph (DAG).
 *
 * Topological ordering implies that children nodes are visited strictly before
 * their parent nodes.
 * The input iterator is invalidated if the corresponding graph changes.
 *
 * @note Topological ordering is not necessarily unique.
 */
class TopoView {
public:
    // value type of TopoView::Iterator
    struct NodeInfo {
        Node* node{nullptr}; // pointer to the current node
        bool isLeaf{false};  // whether the node is a leaf of the subgraph
        bool isRoot{false};  // whether the node is a root of the subgraph
    };

    using NodeSet = std::unordered_set<Node*>;

    // Set of root node pointers.
    struct Roots {
        NodeSet obj;
    };

    // Set of leaf node pointers.
    struct Leaves {
        NodeSet obj;
    };

    /**
     * @brief Create a view on the subgraph spanned by specified roots.
     *
     * @param  roots       the set of root-node pointers
     */
    explicit TopoView(Roots roots)
        : mRoots{std::move(roots.obj)}
    {
    }

    /**
     * @brief Create a view on the subgraph spanned by specified roots and
     * leaves.
     *
     * Leaves that are specified but not reachable from any root will be
     * ignored.
     * Nodes without children which are reachable from a root, without passing
     * through specified leaves, will be automatically considered leaves.
     *
     * @param  roots       the set of root-node pointers
     * @param  leaves      the set of leaf-node pointers
     */
    TopoView(Roots roots, Leaves leaves)
        : mRoots{std::move(roots.obj)}
        , mSpecifiedLeaves{std::move(leaves.obj)}
    {
    }

    ~TopoView() = default;

    TopoView(TopoView const&)                        = default;
    TopoView(TopoView&&) noexcept                    = default;
    auto operator=(TopoView const&) -> TopoView&     = default;
    auto operator=(TopoView&&) noexcept -> TopoView& = default;

    /**
     * @class Iterator
     * @brief Input iterator over a topologically ordered view.
     *
     * Becomes invalid when the corresponding graph is modified.
     * Dereferencing returns a @c NodeInfo object with a raw pointer to the
     * current node.
     *
     * Incrementing throws a @c CyclicGraphError if topological ordering is
     * impossible due to a cycle in the graph.
     */
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = NodeInfo const;
        using pointer           = value_type*;
        using reference         = value_type&;

        /**
         * @brief Create an invalid iterator.
         */
        Iterator() = default;

        /**
         * @brief Create an iterator pointing to the first node of the view.
         *
         * Creates an invalid iterator if the view is empty.
         *
         * @param  view        the topologically ordered view
         */
        explicit Iterator(TopoView const* view)
            : mView{view}
            , mRootsIterator{view->mRoots.cbegin()}
            , mRootsEnd{view->mRoots.cend()}
        {
            findNextValue();
        }

        ~Iterator() = default;

        Iterator(Iterator const&)                        = default;
        Iterator(Iterator&&) noexcept                    = default;
        auto operator=(Iterator const&) -> Iterator&     = default;
        auto operator=(Iterator&&) noexcept -> Iterator& = default;

        /**
         * @brief Returns an info object with pointer to the current node.
         */
        [[nodiscard]] auto operator*() const -> reference
        {
            return mCurrentValue;
        }

        friend auto operator!=(Iterator const& a, Iterator const& b) -> bool
        {
            return a.mPath != b.mPath;
        }

        /**
         * @brief Increment the iterator to the next node in a topological
         * ordering.
         *
         * @throws A @c CyclicGraphError if topological ordering is impossible
         * due to a cycle in the graph.
         */
        auto operator++() -> Iterator&
        {
            mVisited.insert(mPath.head());
            mPath.removeHead();
            findNextValue();
            return *this;
        }

    private:
        void findNextValue()
        {
            // if path is empty, add new root node
            while (mPath.isEmpty() && mRootsIterator != mRootsEnd) {
                Node* root = *(mRootsIterator++);
                if (notYetVisited(root)) {
                    mPath.tryAdd(root);
                    if (isSpecifiedLeaf(root)) {
                        // root is leaf and root (obviously)
                        mCurrentValue = {root, true, true};
                        return;
                    }
                }
            }
            // if path is still empty, iteration ends
            if (mPath.isEmpty()) {
                mCurrentValue = NodeInfo{};
                return;
            }
            // extend path until all children have been visited
            while (true) {
                auto const& next = mPath.next();
                if (next.has_value()) {
                    Node* candidate = next.value();
                    if (notYetVisited(candidate)) {
                        mPath.tryAdd(candidate);
                        if (isSpecifiedLeaf(candidate)) {
                            auto const isRoot = mPath.size() == 1;
                            mCurrentValue     = {candidate, true, isRoot};
                            return;
                        }
                    }
                } else {
                    // head with no children or all its children already visited
                    auto const isLeaf = mPath.head()->children().empty();
                    auto const isRoot = mPath.size() == 1;
                    mCurrentValue     = {mPath.head(), isLeaf, isRoot};
                    return;
                }
            }
        }

        [[nodiscard]] auto notYetVisited(Node* node) -> bool
        {
            return mVisited.find(node) == mVisited.end();
        }

        [[nodiscard]] auto isSpecifiedLeaf(Node* node) -> bool
        {
            return mView->mSpecifiedLeaves.find(node)
                != mView->mSpecifiedLeaves.end();
        }

        TopoView const* mView{nullptr};

        using NodeIterator = typename NodeSet::const_iterator;
        NodeIterator mRootsIterator;
        NodeIterator mRootsEnd;

        Path mPath{};
        NodeSet mVisited; // find visited nodes

        NodeInfo mCurrentValue{};
    };

    [[nodiscard]] static auto end() -> Iterator { return {}; }

    [[nodiscard]] auto begin() const -> Iterator { return Iterator(this); }

private:
    NodeSet mRoots;
    NodeSet mSpecifiedLeaves;
};

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_TOPO_VIEW_HPP
