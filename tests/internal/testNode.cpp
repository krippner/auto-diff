#include "helper/Graph.hpp"

#include <AutoDiff/src/internal/Node.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_container_properties.hpp> // IsEmpty
#include <catch2/matchers/catch_matchers_vector.hpp>               // Equals

using Catch::Matchers::Equals;
using Catch::Matchers::IsEmpty;
using Catch::Matchers::UnorderedEquals;

using AutoDiff::internal::Node;
using AutoDiff::internal::NodeOwner;
using test::root, test::root2, test::x, test::y, test::z, test::u, test::v;

namespace test {

// Implementation testing order of Node destruction
class NodeImpl : public Node {
public:
    NodeImpl(int id, std::vector<int>* dtorSeqPtr)
        : mId{id}
        , mDtorSeqPtr{dtorSeqPtr}
    {
    }

    ~NodeImpl() override
    {
        if (mDtorSeqPtr != nullptr) {
            mDtorSeqPtr->push_back(mId);
        }
    }

    NodeImpl(NodeImpl&&) noexcept                    = default;
    auto operator=(NodeImpl&&) noexcept -> NodeImpl& = default;

    // deleted in base class
    NodeImpl(NodeImpl const&)                    = delete;
    auto operator=(NodeImpl const&) -> NodeImpl& = delete;

private:
    int mId{0};
    std::vector<int>* mDtorSeqPtr{nullptr}; // sequence of destructor calls
};

} // namespace test

using Graph = test::Graph<test::NodeImpl>;

SCENARIO("Single node", "[Node]")
{
    WHEN("creating a root node")
    {
        auto g = Graph(); // no links, all nodes are roots
        THEN("root can be destructed") { CHECK_NOTHROW(g.rootNode().reset()); }
    }
}

SCENARIO("Linear graph", "[Node]")
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

    WHEN("all pointers except root are reset")
    {
        g.zNode().reset();
        g.yNode().reset();
        g.xNode().reset();
        THEN("no node is destructed")
        {
            CHECK_THAT(g.destructorSequence(), IsEmpty());
        }
        THEN("destructing root destructs children in reverse order")
        {
            g.rootNode().reset();
            CHECK_THAT(g.destructorSequence(), Equals<int>({root, z, y, x}));
        }
    }
}

SCENARIO("Diamond graph", "[Node]")
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

    WHEN("all pointers except root are reset")
    {
        g.zNode().reset();
        g.yNode().reset();
        g.xNode().reset();
        THEN("no node is destructed")
        {
            CHECK_THAT(g.destructorSequence(), IsEmpty());
        }
        THEN("destructing root node destructs children in reverse order")
        {
            g.rootNode().reset();
            CHECK_THAT(g.destructorSequence(),
                Equals<int>({root, z, x, y}) || Equals<int>({root, z, y, x}));
        }
    }
}

SCENARIO("Multiple-edge graph", "[Node]")
{
    /**
     * root
     *  ||
     *  x
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.rootNode()->addChild(g.xNode()); // multiple edge, must be ignored

    WHEN("pointer to x is reset")
    {
        g.xNode().reset();
        THEN("no node is destructed")
        {
            CHECK_THAT(g.destructorSequence(), IsEmpty());
        }
        THEN("destructing root node destructs x")
        {
            g.rootNode().reset();
            CHECK_THAT(g.destructorSequence(), Equals<int>({root, x}));
        }
    }
}

SCENARIO("Multi-root graph", "[Node]")
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

    WHEN("all pointers except roots are reset")
    {
        g.vNode().reset();
        g.uNode().reset();
        g.zNode().reset();
        g.yNode().reset();
        g.xNode().reset();
        THEN("no node is destructed")
        {
            CHECK_THAT(g.destructorSequence(), IsEmpty());
        }
        THEN("destructing root2 node only destructs u, v in reverse order")
        {
            g.root2Node().reset();
            CHECK_THAT(g.destructorSequence(), Equals<int>({root2, v, u}));
        }
        THEN("destructing both root node destructs children in reverse order")
        {
            g.root2Node().reset();
            g.rootNode().reset();
            CHECK_THAT(g.destructorSequence(),
                Equals<int>({root2, v, u, root, z, y, x}));
        }
    }
}

SCENARIO("Ownership prevents destruction", "[Node]")
{
    /**
     *  root owner
     *   / \ /
     *  x   y
     *  |   |
     *  u   v
     */

    auto g = Graph();
    g.rootNode()->addChild(g.xNode());
    g.rootNode()->addChild(g.yNode());
    g.xNode()->addChild(g.uNode());
    g.yNode()->addChild(g.vNode());

    WHEN("all pointers except root and y are reset")
    {
        g.xNode().reset();
        g.uNode().reset();
        g.vNode().reset();
        THEN("no node is destructed")
        {
            CHECK_THAT(g.destructorSequence(), IsEmpty());
        }
        THEN("destructing root destructs all nodes except y")
        {
            g.rootNode().reset();
            CHECK_THAT(
                g.destructorSequence(), UnorderedEquals<int>({root, x, u, v}));
        }
        AND_WHEN("registering ownership for y")
        {
            auto const owner = std::make_unique<NodeOwner>();
            g.yNode()->addParentOwner(owner);
            THEN("destructing root does not destruct y and v")
            {
                g.rootNode().reset();
                CHECK_THAT(g.destructorSequence(), Equals<int>({root, u, x}));
            }
        }
        AND_WHEN("registering and then unregistering ownership for y")
        {
            auto const owner = std::make_unique<NodeOwner>();
            g.yNode()->addParentOwner(owner);
            g.yNode()->removeParentOwner(owner);
            THEN("destructing root destructs all nodes except y")
            {
                g.rootNode().reset();
                CHECK_THAT(g.destructorSequence(),
                    UnorderedEquals<int>({root, x, u, v}));
            }
        }
    }
}
