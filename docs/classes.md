# Class diagram

The class diagram below shows a simplified overview of the main AutoDiff classes involved in the following code.

```cpp
Real x, y;
auto expr = x + y;  // Sum<Real, Real>
Real z = var(expr); // Variable<double, double>
Function f(from(x, y), to(z));
f.evaluate();
```

```mermaid
classDiagram
    class Function {
        +evaluate()
        +pushTangent()
        +pullGradient()
        -vector~AbstractComputation*~
    }
    class Variable~Value, Derivative~ {
        +Variable()
        +Variable(Value value)
        +Variable(Operation~Op~ const& op)
        -shared_ptr~Computation~
        -unique_ptr~Owner~
    }
    namespace internal {
        class TopoView {
            +begin()
            +end()
        }
        class Computation~Value, Derivative~ {
            -Value
            -Derivative
            -unique_ptr~AbstractExpression~
        }
        class AbstractComputation {
            +evaluate()
            +pushTangent()
            +pullGradient()
            +setTangentZero(Shape)
            +setGradientZero(Shape)
            +setDerivativeIdentity()
        }
        class Node {
            +~Node()
            +addChild(shared_ptr~Node~ child)
            +releaseChildren()
            +addParentOwner(Owner*)
            +removeParentOwner(Owner*)
            -deleteIteratively(Node*)
        }
        class Expression~Sum~ {
            Sum~Real, Real~ mOp
        }
        class AbstractExpression~Value, Derivative~ {
            +evaluateTo(Value& value)
            +pushForwardTo(Derivative& tangent)
            +pullBack(Derivative const& gradient)
        }
    }
    class Sum~Real, Real~ {
        +valueImpl()
        +pushForwardImpl()
        +pullBackImpl()
    }
    class BinaryOperation~Sum, Real, Real~ {
        + Real x
        + Real y
    }
    class Operation~Sum~ {
        +value()
        +_pushForward()
        +_pullBack()
    }
    Function --> Variable : sources, targets
    Function --> TopoView : compile()
    Function --> AbstractComputation
    TopoView --> Node
    Variable o-- Computation : share
    Computation --|> AbstractComputation : implements
    AbstractComputation --|> Node
    Computation "1" o-- "0..1" AbstractExpression : owns
    Expression --|> AbstractExpression : implements
    Expression *-- Sum
    Sum --|> BinaryOperation
    BinaryOperation --|> Operation
```
