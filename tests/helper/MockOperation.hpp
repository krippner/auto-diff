#ifndef TESTS_HELPER_MOCK_OPERATION_HPP
#define TESTS_HELPER_MOCK_OPERATION_HPP

#include <AutoDiff/src/Core/Expression.hpp>

#include <memory>

namespace test {

/**
 * @brief Operation serving as placeholder in test expressions.
 */
template <typename Value, typename Derivative_>
class MockOperation
    : public AutoDiff::Expression<MockOperation<Value, Derivative_>> {
public:
    // Expression implementation ----------------------------

    using Derivative = Derivative_;

    [[nodiscard]] auto _valueImpl() const -> Value const& { return *mValue; }

    [[nodiscard]] auto _pushForwardImpl() const -> Derivative const&
    {
        return *mDerivative;
    }

    template <typename OtherDerivative>
    void _pullBackImpl(OtherDerivative const& derivative)
    {
        *mDerivative = derivative;
    }

    void _transferChildrenToImpl(AutoDiff::internal::Node& /*node*/) const
    {
        // this is only relevant for computation graphs (see Variable class)
    }

    void _releaseCacheImpl() const
    {
        // no cache to release
    }

    // Test implementation --------------------------------

    auto value() const -> Value& { return *mValue; }
    auto derivative() const -> Derivative& { return *mDerivative; }

private:
    std::shared_ptr<Value> mValue{std::make_shared<Value>()};
    std::shared_ptr<Derivative> mDerivative{std::make_shared<Derivative>()};
};

} // namespace test

#endif // TESTS_HELPER_MOCK_OPERATION_HPP
