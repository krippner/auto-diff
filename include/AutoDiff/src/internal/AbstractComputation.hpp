// Copyright (c) 2024 Matthias Krippner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef AUTODIFF_SRC_INTERNAL_ABSTRACT_COMPUTATION_HPP
#define AUTODIFF_SRC_INTERNAL_ABSTRACT_COMPUTATION_HPP

#include "Node.hpp"
#include "Shape.hpp"

namespace AutoDiff::internal {

/**
 * @class AbstractComputation
 * @brief Abstract base class for computations.
 *
 * A computation is the smallest run-time unit of computation in AutoDiff.
 * It represents a node in the computational graph and computes the value and
 * derivative of a function @a phi associated with a @c Variable.
 *
 * - For literals, @a phi is the identity.
 * - Otherwise,    @a phi is given by an @c Expression of other variables.
 *
 * The derived computation performs three tasks:
 *
 * - Evaluation:      the value of @a phi is computed
 * - Forward-mode AD: a tangent vector is pushed forward by @a phi
 * - Reverse-mode AD: a gradient (covector) is pulled back by @a phi
 *
 * The derived computation is also responsible for caching the results.
 */
class AbstractComputation : public Node {
public:
    /**
     * @brief Compute the value of the associated function @a phi.
     */
    virtual void evaluate() = 0;

#ifndef AUTODIFF_NO_FORWARD_MODE
    /**
     * @brief Set derivative to zero with dimensions matching the tangent.
     *
     * The tangent vector can be viewed as linear map (differential).
     * The shape of its domain is given by the current forward-mode AD.
     * The shape of its codomain is given by the value of @a phi.
     *
     * @param  domainShape the shape of values in the domain
     */
    virtual void setTangentZero(Shape domainShape) = 0;

    /**
     * @brief Compute the pushforward of the current tangent vector.
     *
     * The current tangent vector on the domain of @a phi is mapped to
     * its pushforward on the codomain via the differential of @a phi.
     */
    virtual void pushTangent() = 0;
#endif

#ifndef AUTODIFF_NO_REVERSE_MODE
    /**
     * @brief Set derivative to zero with dimensions matching the gradient.
     *
     * The gradient can be viewed as linear map (differential).
     * The shape of its domain is given by the value of @a phi.
     * The shape of its codomain is given by the current reverse-mode AD.
     *
     * @param  codomainShape the shape of values in the codomain
     */
    virtual void setGradientZero(Shape codomainShape) = 0;

    /**
     * @brief Compute the pullback of the current gradient.
     *
     * The current gradient on the codomain of @a phi is mapped to
     * its pullback on the domain via the differential of @a phi.
     */
    virtual void pullGradient() = 0;
#endif

    /**
     * @brief Set the derivative to the identity map.
     *
     * Propagating the identity map is equivalent to propagating
     * the basis vectors of the co-/domain.
     * The propagated vectors are the rows/columns of the Jacobian matrix.
     */
    virtual void setDerivativeIdentity() = 0;

    /**
     * @brief The shape of the current value of @a phi.
     */
    virtual auto valueShape() const -> Shape = 0;

    /**
     * @brief The shape of the codomain of the current derivative.
     */
    virtual auto derivativeCodomainShape() const -> Shape = 0;
};

} // namespace AutoDiff::internal

#endif // AUTODIFF_SRC_INTERNAL_ABSTRACT_COMPUTATION_HPP
