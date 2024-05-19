// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_EIGEN_ARRAY_OPS_LOG_HPP
#define AUTODIFF_SRC_EIGEN_ARRAY_OPS_LOG_HPP

namespace AutoDiff::EigenAD::Array {

template <typename X>
class Log : public UnaryOperation<Log<X>, X> {
public:
    using Base = UnaryOperation<Log<X>, X>;
    using Base::Base;

    [[nodiscard]] auto _valueImpl() -> decltype(auto)
    {
        return Base::xValue().log();
    }

    [[nodiscard]] auto _pushForwardImpl() -> decltype(auto)
    {
        return Base::xPushForward() / Base::xValue();
    }

    template <typename Derivative>
    void _pullBackImpl(Derivative const& derivative)
    {
        Base::xPullBack(derivative / Base::xValue());
    }
};

} // namespace AutoDiff::EigenAD::Array

namespace AutoDiff {

AUTODIFF_MAKE_ARRAY_UNARY_OP(log, EigenAD::Array::Log)

} // namespace AutoDiff

#endif // AUTODIFF_SRC_EIGEN_ARRAY_OPS_LOG_HPP
