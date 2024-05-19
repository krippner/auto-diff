#include "helper/Graph.hpp"

#include <AutoDiff/src/internal/Node.hpp>
#include <AutoDiff/src/internal/TopoView.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp> // IsEmpty
#include <catch2/matchers/catch_matchers_vector.hpp>               // Equals

using Catch::Matchers::Equals;
using Catch::Matchers::IsEmpty;
using Catch::Matchers::UnorderedEquals;

using test::root, test::root2, test::x, test::y, test::z, test::u, test::v;

namespace test::TopoView {

// Implementation testing iteration over graph
class NodeImpl : public AutoDiff::internal::Node {
public:
    [[nodiscard]] static auto id(Node const* node) -> int
    {
        return dynamic_cast<NodeImpl const*>(node)->mId;
    }

    // needed for test::Graph
    NodeImpl(int id, std::vector<int>* /*dtorSeqPtr*/)
        : mId{id} {};

    ~NodeImpl() override = default;

    NodeImpl(NodeImpl&&) noexcept                    = default;
    auto operator=(NodeImpl&&) noexcept -> NodeImpl& = default;

    // deleted in base class
    NodeImpl(NodeImpl const&)                    = delete;
    auto operator=(NodeImpl const&) -> NodeImpl& = delete;

private:
    int mId{0};
};

} // namespace test::TopoView

using TopoView = AutoDiff::internal::TopoView;
using Graph    = test::Graph<test::TopoView::NodeImpl>;

using AutoDiff::internal::CyclicGraphError;

template <typename... Nodes>
auto rootSet(std::shared_ptr<Nodes>... nodePtrs)
{
    typename TopoView::Roots roots{};
    (roots.obj.insert(nodePtrs.get()), ...);
    return roots;
}

template <typename... Nodes>
auto leafSet(std::shared_ptr<Nodes>... nodePtrs)
{
    typename TopoView::Leaves leaves{};
    (leaves.obj.insert(nodePtrs.get()), ...);
    return leaves;
}

auto getId(TopoView::NodeInfo const& nodeInfo)
{
    return test::TopoView::NodeImpl::id(nodeInfo.node);
}

SCENARIO("Empty graph", "[TopoView]")
{
    WHEN("iterating over empty view")
    {
        auto sequence = std::vector<int>{};
        for (auto nodeInfo : TopoView(rootSet())) {
            sequence.push_back(getId(nodeInfo));
        }
        CHECK_THAT(sequence, IsEmpty());
    }
}

SCENARIO("Single node", "[TopoView]")
{
    auto g = Graph();

    WHEN("iterating over ordered view")
    {
        auto sequence  = std::vector<int>{};
        auto rootFlags = std::vector<int>{};
        auto leafFlags = std::vector<int>{};
        for (auto nodeInfo : TopoView(rootSet(g.rootNode()))) {
            sequence.push_back(getId(nodeInfo));
            if (nodeInfo.isRoot) {
                rootFlags.push_back(getId(nodeInfo));
            }
            if (nodeInfo.isLeaf) {
                leafFlags.push_back(getId(nodeInfo));
            }
        }
        CHECK_THAT(sequence, Equals<int>({root}));
        CHECK_THAT(rootFlags, Equals<int>({root}));
        CHECK_THAT(leafFlags, Equals<int>({root}));
    }
}

SCENARIO("Linear graph", "[TopoView]")
{
    /**
     * root
     *  |
     *  x
     *  |
     *  y
     *  |
     *  z
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.xNode()->addChild(g.yNode());
    g.yNode()->addChild(g.zNode());

    WHEN("iterating over ordered view")
    {
        auto sequence  = std::vector<int>{};
        auto rootFlags = std::vector<int>{};
        auto leafFlags = std::vector<int>{};
        for (auto nodeInfo : TopoView(rootSet(g.rootNode()))) {
            sequence.push_back(getId(nodeInfo));
            if (nodeInfo.isRoot) {
                rootFlags.push_back(getId(nodeInfo));
            }
            if (nodeInfo.isLeaf) {
                leafFlags.push_back(getId(nodeInfo));
            }
        }
        CHECK_THAT(sequence, Equals<int>({z, y, x, root}));
        CHECK_THAT(rootFlags, Equals<int>({root}));
        CHECK_THAT(leafFlags, Equals<int>({z}));
    }
}

SCENARIO("Tree graph", "[TopoView]")
{
    /**
     * root
     *  /\
     * x  y
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.rootNode()->addChild(g.yNode());

    auto const view = TopoView(rootSet(g.rootNode()));

    WHEN("iterating over ordered view")
    {
        auto sequence  = std::vector<int>{};
        auto rootFlags = std::vector<int>{};
        auto leafFlags = std::vector<int>{};
        for (auto nodeInfo : view) {
            sequence.push_back(getId(nodeInfo));
            if (nodeInfo.isRoot) {
                rootFlags.push_back(getId(nodeInfo));
            }
            if (nodeInfo.isLeaf) {
                leafFlags.push_back(getId(nodeInfo));
            }
        }
        CHECK_THAT(
            sequence, Equals<int>({x, y, root}) || Equals<int>({y, x, root}));
        CHECK_THAT(rootFlags, Equals<int>({root}));
        CHECK_THAT(leafFlags, UnorderedEquals<int>({x, y}));
    }
}

SCENARIO("Diamond graph", "[TopoView]")
{
    /**
     * root
     *  /\
     * x  y
     *  \/
     *  z
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.rootNode()->addChild(g.yNode());
    g.xNode()->addChild(g.zNode());
    g.yNode()->addChild(g.zNode());

    WHEN("iterating over ordered view")
    {
        auto sequence  = std::vector<int>{};
        auto rootFlags = std::vector<int>{};
        auto leafFlags = std::vector<int>{};
        for (auto nodeInfo : TopoView(rootSet(g.rootNode()))) {
            sequence.push_back(getId(nodeInfo));
            if (nodeInfo.isRoot) {
                rootFlags.push_back(getId(nodeInfo));
            }
            if (nodeInfo.isLeaf) {
                leafFlags.push_back(getId(nodeInfo));
            }
        }
        CHECK_THAT(sequence,
            Equals<int>({z, x, y, root}) || Equals<int>({z, y, x, root}));
        CHECK_THAT(rootFlags, Equals<int>({root}));
        CHECK_THAT(leafFlags, Equals<int>({z}));
    }
}

SCENARIO("Multiple-edge graph", "[TopoView]")
{
    /**
     * root
     *  ||
     *  x
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.rootNode()->addChild(g.xNode());

    WHEN("iterating over ordered view")
    {
        auto sequence  = std::vector<int>{};
        auto rootFlags = std::vector<int>{};
        auto leafFlags = std::vector<int>{};
        for (auto nodeInfo : TopoView(rootSet(g.rootNode()))) {
            sequence.push_back(getId(nodeInfo));
            if (nodeInfo.isRoot) {
                rootFlags.push_back(getId(nodeInfo));
            }
            if (nodeInfo.isLeaf) {
                leafFlags.push_back(getId(nodeInfo));
            }
        }
        CHECK_THAT(sequence, Equals<int>({x, root}));
        CHECK_THAT(rootFlags, Equals<int>({root}));
        CHECK_THAT(leafFlags, Equals<int>({x}));
    }
}

SCENARIO("Self-referencing root", "[TopoView]")
{
    /**
     * root <-
     *  | \__/
     *  x
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.rootNode()->addChild(g.rootNode());

    THEN("iterating over ordered view throws exception")
    {
        auto const view      = TopoView(rootSet(g.rootNode()));
        auto const iteration = [&]() {
            for (auto nodeInfo : view) {
                (void)nodeInfo;
            }
        };
        CHECK_THROWS_AS(iteration(), CyclicGraphError);
    }
}

SCENARIO("Graph cycle at root", "[TopoView]")
{
    /**
     * root <-
     *  |     \
     *  x      |
     *  |     /
     *  y ----
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.xNode()->addChild(g.yNode());
    g.yNode()->addChild(g.rootNode());

    THEN("iterating over ordered view throws exception")
    {
        auto const view      = TopoView(rootSet(g.rootNode()));
        auto const iteration = [&]() {
            for (auto nodeInfo : view) {
                (void)nodeInfo;
            }
        };
        CHECK_THROWS_AS(iteration(), CyclicGraphError);
    }
}

SCENARIO("Graph cycle below root", "[TopoView]")
{
    /**
     * root
     *  |
     *  x <-
     *  |   \
     *  y    |
     *  |   /
     *  z --
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.xNode()->addChild(g.yNode());
    g.yNode()->addChild(g.zNode());
    g.zNode()->addChild(g.xNode());

    THEN("accessing ordered view throws exception")
    {
        auto const view      = TopoView(rootSet(g.rootNode()));
        auto const iteration = [&]() {
            for (auto nodeInfo : view) {
                (void)nodeInfo;
            }
        };
        CHECK_THROWS_AS(iteration(), CyclicGraphError);
    }
}

SCENARIO("Multi-root graph", "[TopoView]")
{
    /**
     *  root root2
     *     | /|
     *     |/ |
     *     x  u
     *     | /
     *     |/
     *     y
     */

    auto g = Graph();
    WHEN("adding x before u")
    {
        g.rootNode()->addChild(g.xNode());
        g.root2Node()->addChild(g.xNode());
        g.root2Node()->addChild(g.uNode());
        g.xNode()->addChild(g.yNode());
        g.uNode()->addChild(g.yNode());

        WHEN("iterating over ordered view")
        {
            auto sequence  = std::vector<int>{};
            auto rootFlags = std::vector<int>{};
            auto leafFlags = std::vector<int>{};
            for (auto nodeInfo :
                TopoView(rootSet(g.rootNode(), g.root2Node()))) {
                sequence.push_back(getId(nodeInfo));
                if (nodeInfo.isRoot) {
                    rootFlags.push_back(getId(nodeInfo));
                }
                if (nodeInfo.isLeaf) {
                    leafFlags.push_back(getId(nodeInfo));
                }
            }
            CHECK_THAT(sequence, Equals<int>({y, x, root, u, root2})
                                     || Equals<int>({y, u, x, root2, root})
                                     || Equals<int>({y, x, u, root2, root}));
            CHECK_THAT(rootFlags, UnorderedEquals<int>({root, root2}));
            CHECK_THAT(leafFlags, Equals<int>({y}));
        }
    }
    WHEN("adding u before x")
    {
        g.rootNode()->addChild(g.xNode());
        g.root2Node()->addChild(g.uNode());
        g.root2Node()->addChild(g.xNode());
        g.xNode()->addChild(g.yNode());
        g.uNode()->addChild(g.yNode());

        WHEN("iterating over ordered view")
        {
            auto sequence  = std::vector<int>{};
            auto rootFlags = std::vector<int>{};
            auto leafFlags = std::vector<int>{};
            for (auto nodeInfo :
                TopoView(rootSet(g.rootNode(), g.root2Node()))) {
                sequence.push_back(getId(nodeInfo));
                if (nodeInfo.isRoot) {
                    rootFlags.push_back(getId(nodeInfo));
                }
                if (nodeInfo.isLeaf) {
                    leafFlags.push_back(getId(nodeInfo));
                }
            }
            CHECK_THAT(sequence, Equals<int>({y, x, root, u, root2})
                                     || Equals<int>({y, u, x, root2, root})
                                     || Equals<int>({y, x, u, root2, root}));
            CHECK_THAT(rootFlags, UnorderedEquals<int>({root, root2}));
            CHECK_THAT(leafFlags, Equals<int>({y}));
        }
    }
}

SCENARIO("View of subgraph", "[TopoView]")
{
    /**
     *  root root2
     *     | /|
     *     |/ |
     *     x  u
     *     | /|
     *     |/ |
     *     y  v
     *     | /
     *     |/
     *     z
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.root2Node()->addChild(g.xNode());
    g.root2Node()->addChild(g.uNode());
    g.xNode()->addChild(g.yNode());
    g.uNode()->addChild(g.yNode());
    g.uNode()->addChild(g.vNode());
    g.yNode()->addChild(g.zNode());
    g.vNode()->addChild(g.zNode());

    auto const roots = rootSet(g.rootNode(), g.root2Node());

    WHEN("viewing with leaves = {u, y}")
    {
        auto const leaves = leafSet(g.uNode(), g.yNode());
        auto sequence     = std::vector<int>{};
        auto rootFlags    = std::vector<int>{};
        auto leafFlags    = std::vector<int>{};
        for (auto nodeInfo : TopoView(roots, leaves)) {
            sequence.push_back(getId(nodeInfo));
            if (nodeInfo.isRoot) {
                rootFlags.push_back(getId(nodeInfo));
            }
            if (nodeInfo.isLeaf) {
                leafFlags.push_back(getId(nodeInfo));
            }
        }
        CHECK_THAT(sequence, Equals<int>({y, x, root, u, root2})
                                 || Equals<int>({u, y, x, root2, root})
                                 || Equals<int>({y, x, u, root2, root}));
        CHECK_THAT(rootFlags, UnorderedEquals<int>({root, root2}));
        CHECK_THAT(leafFlags, UnorderedEquals<int>({u, y}));
    }
    WHEN("viewing with leaves = {root, root2}")
    {
        auto const leaves = leafSet(g.rootNode(), g.root2Node());
        auto sequence     = std::vector<int>{};
        auto rootFlags    = std::vector<int>{};
        auto leafFlags    = std::vector<int>{};
        for (auto nodeInfo : TopoView(roots, leaves)) {
            sequence.push_back(getId(nodeInfo));
            if (nodeInfo.isRoot) {
                rootFlags.push_back(getId(nodeInfo));
            }
            if (nodeInfo.isLeaf) {
                leafFlags.push_back(getId(nodeInfo));
            }
        }
        CHECK_THAT(sequence, UnorderedEquals<int>({root, root2}));
        CHECK_THAT(rootFlags, UnorderedEquals<int>({root, root2}));
        CHECK_THAT(leafFlags, UnorderedEquals<int>({root, root2}));
    }
    WHEN("viewing with leaves = {u}")
    {
        auto const leaves = leafSet(g.uNode());
        auto sequence     = std::vector<int>{};
        auto rootFlags    = std::vector<int>{};
        auto leafFlags    = std::vector<int>{};
        for (auto nodeInfo : TopoView(roots, leaves)) {
            sequence.push_back(getId(nodeInfo));
            if (nodeInfo.isRoot) {
                rootFlags.push_back(getId(nodeInfo));
            }
            if (nodeInfo.isLeaf) {
                leafFlags.push_back(getId(nodeInfo));
            }
        }
        CHECK_THAT(sequence, Equals<int>({z, y, x, root, u, root2})
                                 || Equals<int>({z, y, u, x, root2, root})
                                 || Equals<int>({z, y, x, u, root2, root})
                                 || Equals<int>({u, z, y, x, root2, root}));
        CHECK_THAT(rootFlags, UnorderedEquals<int>({root, root2}));
        CHECK_THAT(leafFlags, UnorderedEquals<int>({u, z}));
    }
}

SCENARIO("Iterating over view has no side effects", "[TopoView]")
{
    /**
     * root
     *  /\
     * x  y
     *  \/
     *  z
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.rootNode()->addChild(g.yNode());
    g.xNode()->addChild(g.zNode());
    g.yNode()->addChild(g.zNode());

    WHEN("iterating over separate views in separate scopes")
    {
        auto sequence = std::vector<int>{};
        for (auto nodeInfo : TopoView(rootSet(g.rootNode()))) {
            sequence.push_back(getId(nodeInfo));
        }
        sequence.clear();
        for (auto nodeInfo : TopoView(rootSet(g.rootNode()))) {
            sequence.push_back(getId(nodeInfo));
        }
        CHECK_THAT(sequence,
            Equals<int>({z, x, y, root}) || Equals<int>({z, y, x, root}));
    }
    WHEN("iterating over separate views in same scope")
    {
        auto const view1 = TopoView(rootSet(g.rootNode()));
        auto const view2 = TopoView(rootSet(g.rootNode()));
        auto sequence    = std::vector<int>{};
        for (auto nodeInfo : view1) {
            sequence.push_back(getId(nodeInfo));
        }
        sequence.clear();
        for (auto nodeInfo : view2) {
            sequence.push_back(getId(nodeInfo));
        }
        CHECK_THAT(sequence,
            Equals<int>({z, x, y, root}) || Equals<int>({z, y, x, root}));
    }
    WHEN("iterating over same view multiple times")
    {
        auto const view = TopoView(rootSet(g.rootNode()));
        auto sequence   = std::vector<int>{};
        for (auto nodeInfo : view) {
            sequence.push_back(getId(nodeInfo));
        }
        sequence.clear();
        for (auto nodeInfo : view) {
            sequence.push_back(getId(nodeInfo));
        }
        CHECK_THAT(sequence,
            Equals<int>({z, x, y, root}) || Equals<int>({z, y, x, root}));
    }
}
