#ifndef TESTS_INTERNAL_HELPER_GRAPH_HPP
#define TESTS_INTERNAL_HELPER_GRAPH_HPP

#include <memory>
#include <vector>

namespace test {

// Node ids
enum { root, root2, x, y, z, u, v };

// Simplify node creation and setup
template <typename Impl>
class Graph {
public:
    using ImplPtr = std::shared_ptr<Impl>;

    Graph()
        : mDtorSeq{}
        , mRootNode{std::make_shared<Impl>(root, &mDtorSeq)}
        , mRoot2Node{std::make_shared<Impl>(root2, &mDtorSeq)}
        , mXNode{std::make_shared<Impl>(x, &mDtorSeq)}
        , mYNode{std::make_shared<Impl>(y, &mDtorSeq)}
        , mZNode{std::make_shared<Impl>(z, &mDtorSeq)}
        , mUNode{std::make_shared<Impl>(u, &mDtorSeq)}
        , mVNode{std::make_shared<Impl>(v, &mDtorSeq)}
    {
    }

    auto rootNode() -> ImplPtr& { return mRootNode; }
    auto root2Node() -> ImplPtr& { return mRoot2Node; }
    auto xNode() -> ImplPtr& { return mXNode; }
    auto yNode() -> ImplPtr& { return mYNode; }
    auto zNode() -> ImplPtr& { return mZNode; }
    auto uNode() -> ImplPtr& { return mUNode; }
    auto vNode() -> ImplPtr& { return mVNode; }

    [[nodiscard]] auto destructorSequence() const -> std::vector<int> const&
    {
        return mDtorSeq;
    }

    void clearDestructorSequence() { mDtorSeq.clear(); }

private:
    std::vector<int> mDtorSeq; // sequence of destructor calls
    ImplPtr mRootNode;
    ImplPtr mRoot2Node;
    ImplPtr mXNode;
    ImplPtr mYNode;
    ImplPtr mZNode;
    ImplPtr mUNode;
    ImplPtr mVNode;
};

} // namespace test

#endif // TESTS_INTERNAL_HELPER_GRAPH_HPP
