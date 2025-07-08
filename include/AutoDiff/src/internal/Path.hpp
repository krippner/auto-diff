// Copyright (c) 2024-2025 Matthias Krippner
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

    // path must not be empty
    [[nodiscard]] auto tail() const -> Node*

    // path must not be empty
    void removeTail()
    {
        mSet.erase(tail());
        mStack.pop();
    }

    // path must not be empty
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
