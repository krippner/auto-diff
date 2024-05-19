/**
 * @brief Demo of reverse-mode automatic differentiation (aka backpropagation).
 *
 * This demo trains a neural network on a dataset to solve the XOR problem,
 * which is a classic example in neural network theory used to illustrate
 * the need for hidden layers.
 *
 * The neural network consists of one hidden layer and an output layer.
 * The training process adjusts the network parameters (weights)
 * to minimize the mean squared error (MSE) between the network output and
 * the expected output, which is the XOR of the two inputs.
 *
 * The AutoDiff library is used to compute the gradient of the loss function
 * with respect to the network parameters.
 * This gradient is then used to update the parameters,
 * using backtracking line search for optimal step size (learning rate).
 *
 * Copyright (c) 2024 Matthias Krippner
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

// This module enables automatic differentiation for Eigen types.
#include <AutoDiff/Eigen>

// Speed up compilation and produce smaller binary:
// Backpropagation only needs reverse-mode automatic differentiation.
// We can remove all code related to forward-mode by defining
// the following macro **before** including <AutoDiff/Core>.
#define AUTODIFF_NO_FORWARD_MODE

// The core framework provides the classes 'Variable' and 'Function'.
#include <AutoDiff/Core>

// Include the implementation for relevant Eigen types.
#include <Eigen/Core>

#include <cmath> // sqrt
#include <cstdio>
#include <iostream>
#include <utility> // pair
#include <vector>

// Bring common symbols into the current namespace.
using AutoDiff::Function;
using AutoDiff::var;

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

using std::cout;
using std::printf;

// Define a dataset as a vector of input-output pairs.
using Dataset = std::vector<std::pair<Vector2d, double>>;

/**
 * @brief Sigmoid activation function for use in the network.
 *
 * @note Pass AutoDiff expressions by const reference to avoid copies.
 *
 * @param input An expression computing the input to the artificial neuron
 */
template <typename Expr>
auto sigmoid(AutoDiff::Expression<Expr> const& input)
{
    // This expression uses NumPy-style broadcasting
    // to apply the sigmoid to the input coefficient-wise.
    return 1 / (1 + exp(-4 * input));
}

int main()
{
    printf("## Training a XOR network using backpropagation ##\n");

    printf("Setting up neural network...\n");

    // Define network parameters.
    Index const inputSize  = 2 + 1; // 2d input + bias
    Index const hiddenSize = 3;
    MatrixXd hiddenWeights
        = MatrixXd::Random(hiddenSize, inputSize) / inputSize;
    VectorXd outputWeights = VectorXd::Random(hiddenSize) / hiddenSize;
    // Allocate the gradients.
    RowVectorXd hiddenGradient(hiddenWeights.size());
    RowVectorXd outputGradient(outputWeights.size());

    // Define AutoDiff variables from network parameters.
    auto const hiddenWeightsVar = var(hiddenWeights);
    auto const outputWeightsVar = var(outputWeights);
    auto const inputVar         = var(Vector3d::Ones());
    auto const targetVar        = var(0.0);
    // Define network output and loss function as AutoDiff variables.
    auto const hidden         = sigmoid(hiddenWeightsVar * inputVar);
    auto const outputVar      = var(sigmoid(dot(outputWeightsVar, hidden)));
    auto const squaredLossVar = var(square(targetVar - outputVar));

    // Define the network as a differentiable (AutoDiff) function from
    // the input and weight variables (sources) to the squared loss (target).
    // The function sources don't need to be specified here.
    Function network(squaredLossVar);

    // Training parameters
    auto const threshold = 0.001; // determine whether training converged
    auto const maxEpochs = 1000;  // terminate even if it did not converge
    auto const maxIters  = 10;    // max number of iterations for line search

    Dataset const data = {
        {Vector2d(0, 0), 0},     // (0,0) ↦ 0
        {Vector2d(1, 0), 1},     // (1,0) ↦ 1
        {Vector2d(0, 1), 1},     // (0,1) ↦ 1
        {Vector2d(1, 1), 0},     // (1,1) ↦ 0
        {Vector2d(0.5, 0.5), 0}, // | more samples
        {Vector2d(2, 0), 1},     // | to better avoid
        {Vector2d(0, 2), 1}      // | local minima
    };

    printf("Starting training...");
    auto currentMSE   = 0.0;
    auto learningRate = 1.0;
    auto epoch        = 0;
    auto hasConverged = false;
    while (epoch++ < maxEpochs) {
        // A) Find descent direction (batch learning)
        // ------------------------------------------
        currentMSE = 0.0;
        hiddenGradient.setZero();
        outputGradient.setZero();
        for (auto const& sample : data) {
            // Augment the 2d input with a dummy input for the bias.
            inputVar  = Vector3d(sample.first(0), sample.first(1), 1);
            targetVar = sample.second;

            // Evaluate the network for the current sample.
            network.evaluate();
            // The operator() returns the value of an AutoDiff variable.
            currentMSE += squaredLossVar();

            // Compute the gradient(s) of the current loss (backpropagation).
            network.pullGradientAt(squaredLossVar);
            // The gradient of the sum over all samples is
            // the sum of the gradients for each sample.
            hiddenGradient += d(hiddenWeightsVar);
            outputGradient += d(outputWeightsVar);
        }
        currentMSE /= data.size();

        // Check for convergence.
        auto gradientNorm = std::sqrt(
            hiddenGradient.squaredNorm() + outputGradient.squaredNorm());
        if (gradientNorm < threshold) {
            hasConverged = true;
            break;
        }
        // Normalize the gradient to get the descent direction.
        hiddenGradient /= gradientNorm;
        outputGradient /= gradientNorm;

        // B) Optimize learning rate by line search
        // ----------------------------------------
        // Try larger learning rate first.
        learningRate *= 1.5;
        for (auto iter = 0; iter < maxIters; ++iter) {
            // Gradient descent step:
            // Updates weights in descent direction starting from the
            // previously cached weights.
            // In AutoDiff, gradients w.r.t. matrix variables are flattened
            // to row vectors. We need to reshape them back to matrices.
            hiddenWeightsVar
                = hiddenWeights
                - learningRate * hiddenGradient.reshaped(hiddenSize, inputSize);
            outputWeightsVar
                = outputWeights - learningRate * outputGradient.transpose();

            // Compute next MSE with current learning rate.
            double nextMSE = 0.0;
            for (auto const& sample : data) {
                inputVar  = Vector3d(sample.first(0), sample.first(1), 1);
                targetVar = sample.second;

                network.evaluate();
                nextMSE += squaredLossVar();
            }
            nextMSE /= data.size();

            // Check success using Armijo condition.
            auto const c = 0.1;
            if (currentMSE - nextMSE > learningRate * c * gradientNorm) {
                // Accept the step and cache the new weights.
                hiddenWeights = hiddenWeightsVar();
                outputWeights = outputWeightsVar();
                break;
            }
            // else, learning rate too large → backtrack
            auto const tau = 0.8;
            learningRate *= tau;
        }
        // printf("Epoch %d: MSE = %f, learning rate = %f, |gradient| = %f\n",
        //     epoch, currentMSE, learningRate, gradientNorm);
    }
    printf(" finished after %d epochs with MSE = %f.\n", epoch, currentMSE);
    if (!hasConverged) {
        printf("Warning: training did not converge.\n");
    }
    cout << "Hidden weights:\n" << hiddenWeights << "\n";
    cout << "Output weights:\n" << outputWeights << "\n";

    printf("Evaluating trained network...\n");
    std::vector<Vector2d> const evaluationSet = {
        Vector2d(0, 0),     //
        Vector2d(1, 0),     //
        Vector2d(0, 1),     //
        Vector2d(1, 1),     //
        Vector2d(0.5, 0.5), //
        Vector2d(0, 0.5),   //
        Vector2d(0.5, 0),   //
        Vector2d(1, 0.5),   //
        Vector2d(0.5, 1),   //
        Vector2d(0.5, 2),   //
        Vector2d(2, 0.5)    //
    };
    for (auto const& input : evaluationSet) {
        inputVar = Vector3d(input(0), input(1), 1);
        network.evaluate();
        cout << "(" << input.transpose() << ") ↦ " << outputVar() << "\n";
    }
}
