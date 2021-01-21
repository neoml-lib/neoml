# CELULayer Class

<!-- TOC -->

- [CELULayer Class](#celulayer-class)
    - [Settings](#settings)
        - [Multiplier](#multiplier)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates the `ELU` activation function for each element of a single input.

The activation function formula:

```c++
f(x) = GetAlpha() * (exp(x) - 1)    if x < 0
f(x) = x                            if x >= 0
```

## Settings

### Multiplier

```c++
void SetAlpha( float alpha );
```

Sets the multiplier before the exponential function used for negative values of `x`.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

There is only one input accepting a data blob of arbitrary size.

## Outputs

There is only one output, which contains a blob of the same size as the input; each element of the output is the value of the activation function for the corresponding element of the input. 
