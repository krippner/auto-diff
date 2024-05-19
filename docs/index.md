# Documentation

## Core framework

The _Core_ framework of _AutoDiff_ implements the abstract coordinate-free concepts of differentiation (see [Mathematical background](#mathematical-background) below).
It offers class templates for evaluating and differentiating expressions of any value and derivative type.
Through a plugin system, separate modules can provide concrete implementation of these concepts for particular combinations of value and derivative types.

- [Variables](core/variable.md)
  - [Creating variables from values](core/variable.md#creating-variables-from-values)
  - [Variable getters and setters](core/variable.md#variable-getters-and-setters)
  - [Copying variables](core/variable.md#copying-variables)
- [Expressions](core/expression.md)
  - [Evaluating expressions](core/expression.md#evaluating-expressions)
  - [Custom expressions](core/expression.md#custom-expressions)
  - [Memory management in expressions](core/expression.md#memory-management-in-expressions)
- [Functions](core/function.md)
  - [Defining functions](core/function.md#defining-functions)
  - [Evaluating functions](core/function.md#evaluating-functions)
  - [Forward-mode differentiation](core/function.md#forward-mode-differentiation)
  - [Reverse-mode differentiation (aka backpropagation)](core/function.md#reverse-mode-differentiation-aka-backpropagation)
  - [Function compilation](core/function.md#function-compilation)

## Modules

AutoDiff is designed as a framework that can be extended by users with additional modules for specific value and derivative types.
The following modules are currently available as part of the library:

- [Basic module](modules/basic.md) - scalar expressions with C++ built-in types
- [Eigen module](modules/eigen.md) - expressions with scalars, Eigen arrays, and (dense) Eigen matrices

## Applications

- [Applications](applications.md)
  - [Control flow](applications.md#control-flow)
  - [Computing the Jacobian matrix](applications.md#computing-the-jacobian-matrix)
  - [Gradient computation](applications.md#gradient-computation)
  - [Element-wise gradient computation](applications.md#element-wise-gradient-computation)
  - [Jacobian-vector products](applications.md#jacobian-vector-products)

## Mathematical background

- [Mathematical notes on the API](math/api-math.md)
- [Automatic differentiation and differential geometry](math/diff-geo.md)
- [Differentiation in local coordinates](math/diff-geo-chart.md)

## Advanced topics

- [Developer guide](developer.md)
- [Class diagram](classes.md)
