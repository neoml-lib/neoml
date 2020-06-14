# CReLULayer Class

<!-- TOC -->

- [CReLULayer Class](#crelulayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates the `ReLU` activation function for each element of a single input.

Here is the default formula of the activation function:

```c++
f(x) = 0    if x <= 0
f(x) = x    if x > 0
```

You also can set the cutoff upper threshold for the function. If you do, the function will be calculated according to the formula:

```c++
f(x) = 0            if x <= 0
f(x) = x            if 0 < x < threshold
f(x) = threshold    if threshold <= x
```

## Settings

```c++
void SetUpperThreshold( float threshold );
```

Sets the upper threshold for the value of the function.
By default there is no threshold, the function is not bounded from above.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

There is only one input, which accepts a [blob](../DnnBlob.md) of any size.


## Outputs

There is only one output, which returns a blob of the same size as the input blob. Each element of the output contains the value of the activation function calculated on the corresponding element of the input.