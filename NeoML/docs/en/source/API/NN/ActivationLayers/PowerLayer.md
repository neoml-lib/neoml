# CPowerLayer Class

<!-- TOC -->

- [CPowerLayer Class](#cpowerlayer-class)
    - [Settings](#settings)
        - [Exponent](#exponent)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that raises each element of the input to the given power.

The activation function is calculated according to the formula:

```c++
f(x) = pow(x, GetExponent())
```

## Settings

### Exponent

```c++
void SetExponent( float exponent );
```

Sets the power to which the input elements will be raised.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

There is only one input, which accepts a data blob of arbitrary size.

## Outputs

There is only one output, which returns a blob of the same size as the input blob. Each element of the output contains the value of the activation function calculated on the corresponding element of the input.
