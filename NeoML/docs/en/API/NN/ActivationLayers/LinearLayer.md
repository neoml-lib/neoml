# CLinearLayer Class

<!-- TOC -->

- [CLinearLayer Class](#clinearlayer-class)
    - [Settings](#settings)
        - [Multiplier](#multiplier)
        - [Free term](#free-term)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates a linear activation function for each element of a single input.

The activation function is calculated according to the formula:

```c++
f(x) = GetMultiplier() * x + GetFreeTerm()
```

## Settings

### Multiplier

```c++
void SetMultiplier( float multiplier );
```

Sets the multiplier value.

### Free term

```c++
void SetFreeTerm( float freeTerm );
```

Sets the free term value.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

There is only one input, which accepts a data blob of arbitrary size.

## Outputs

There is only one output, which returns a blob of the same size as the input blob. Each element of the output contains the value of the activation function calculated on the corresponding element of the input.
