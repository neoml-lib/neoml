# CHardTanhLayer Class

<!-- TOC -->

- [CHardTanhLayer Class](#chardtanhlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates the `HardTanh` activation function for each element of a single input.

The activation function has the following formula:

```c++
f(x) = -1    if x <= -1
f(x) = x     if -1 < x < 1
f(x) = 1     if x >= 1
```

## Settings

There are no settings for this layer.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

There is only one input, which accepts a data blob of arbitrary size.

## Outputs

There is only one output of the same size as the input. Each element contains the value of the activation function on the corresponding element of the input.
