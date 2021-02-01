# CGELULayer Class

<!-- TOC -->

- [CGELULayer Class](#cgelulayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates the `GELU` activation function for each element of a single input.

The activation function formula:

```c++
f(x) = x * sigmoid( 1.702 * x )
```

## Settings

There are no settings for this layer.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

There is only one input, which accepts a data blob of arbitrary size.

## Outputs

There is only one output, which returns a blob of the same size as the input blob. Each element of the output contains the value of the activation function calculated on the corresponding element of the input.
