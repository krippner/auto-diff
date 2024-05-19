// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_INTERNAL_PATH_HPP
#define AUTODIFF_SRC_INTERNAL_PATH_HPP

#include "Node.hpp"

#include <optional>
#include <stack>
#include <stdexcept> // logic_error
#include <unordered_set>

namespace AutoDiff::internal {

/**
 * @class CyclicGraphError
 * @brief Exception thrown when a graph cycle is detected.
 */
class CyclicGraphError : public std::logic_error {
public:
    CyclicGraphError()
        : logic_error("Graph is cyclic.")
    {
    }
};

/**
 * @class Path
 * @brief Represents a path through a graph.
 *
 * The path is represented as a stack of nodes and keeps track of nodes
 * that have been visited.
 */
class Path {
public:
    friend auto operator!=(Path const& a, Path const& b) -> bool
    {
        return a.mStack.size() != b.mStack.size();
    }

    void tryAdd(Node* node)
    {
        auto const [_, inserted] = mSet.insert(node);
        if (!inserted) {
            throw CyclicGraphError();
        }
        mStack.push({node, node->children().cbegin()});
    }

    [[nodiscard]] auto isEmpty() const -> bool { return mStack.empty(); }

    [[nodiscard]] auto size() const -> std::size_t { return mStack.size(); }

    // causes undefined behaviour if path is empty
    [[nodiscard]] auto head() -> Node* { return mStack.top().node; }

    // causes undefined behaviour if path is empty
    void removeHead()
    {
        mSet.erase(head());
        mStack.pop();
    }

    // causes undefined behaviour if path is empty
    auto next() -> std::optional<Node*>
    {
        auto& current = mStack.top();
        if (current.childIter == current.node->children().cend()) {
            return std::nullopt;
        }
        return (current.childIter++)->get();
    }

private:
    struct Element {
        Node* node{nullptr};
        typename Node::PtrSet::const_iterator childIter;
    };

    std::stack<Element> mStack;
    std::unordered_set<Node const*> mSet; // find nodes on path
};

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_PATH_HPP
