#ifndef TESTS_CORE_HELPER_IDENTITY_HPP
#define TESTS_CORE_HELPER_IDENTITY_HPP

#include <AutoDiff/src/Core/UnaryOperation.hpp>

#include <vector>

namespace test {

/**
 * @brief Simple implementation of a unary operation.
 *
 * @tparam X           the type of the operand
 */
template <typename X>
class Identity : public AutoDiff::UnaryOperation<Identity<X>, X> {
public:
    using Base = AutoDiff::UnaryOperation<Identity<X>, X>;
    using Base::Base;

    auto _valueImpl() -> decltype(auto) { return this->xValue(); }

    auto _pushForwardImpl() -> decltype(auto) { return this->xPushForward(); }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        this->xPullBack(derivative);
    }
};

template <typename X>
auto identity(AutoDiff::Expression<X> const& x) -> decltype(auto)
{
    return Identity<X>{x};
}

template <typename X>
class IdentityWithId : public AutoDiff::UnaryOperation<IdentityWithId<X>, X> {
public:
    using Base = AutoDiff::UnaryOperation<IdentityWithId<X>, X>;

    IdentityWithId(
        AutoDiff::Expression<X> const& x, int id, std::vector<int>* dtorSeqPtr)
        : Base{x.derived()}
        , mId{id}
        , mDtorSeqPtr{dtorSeqPtr}
    {
    }

    ~IdentityWithId() { mDtorSeqPtr->push_back(mId); }

    IdentityWithId(IdentityWithId const&)                        = default;
    IdentityWithId(IdentityWithId&&) noexcept                    = default;
    auto operator=(IdentityWithId const&) -> IdentityWithId&     = default;
    auto operator=(IdentityWithId&&) noexcept -> IdentityWithId& = default;

    auto _valueImpl() -> decltype(auto) { return this->xValue(); }

    auto _pushForwardImpl() -> decltype(auto) { return this->xPushForward(); }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        this->xPullBack(derivative);
    }

private:
    int mId{};
    std::vector<int>* mDtorSeqPtr{nullptr};
};

} // namespace test

#endif // TESTS_CORE_HELPER_IDENTITY_HPP
