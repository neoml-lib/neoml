# CLeakyReLULayer Class

<!-- TOC -->

- [CLeakyReLULayer Class](#cleakyrelulayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates the `LeakyReLU` activation function for each element of a single input.

Here is the formula of the activation function:

```c++
f(x) = alpha * x    if x <= 0
f(x) = x            if x > 0
```

## Settings

```c++
void SetAlpha( float alpha );
```

Sets the multiplier used for negative values of `x`. It is equal to `0` by default, which makes the function equivalent to `ReLU`.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

There is only one input, which accepts a data blob of arbitrary size.

## Outputs

There is only one output, which returns a blob of the same size as the input blob. Each element of the output contains the value of the activation function calculated on the corresponding element of the input.
