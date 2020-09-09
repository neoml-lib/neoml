# CHSwishLayer Class

<!-- TOC -->

- [CHSwishLayer Class](#chswishlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates the `h-swish` activation function for each element of a single input.

The activation function formula:

```c++
f(x) = x * ReLU6( x + 3 ) / 6
```

## Settings

There are no settings for this layer.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

There is only one input, which accepts a data blob of arbitrary size.

## Outputs

There is only one output, which returns a blob of the same size as the input blob. Each element of the output contains the value of the activation function calculated on the corresponding element of the input.
