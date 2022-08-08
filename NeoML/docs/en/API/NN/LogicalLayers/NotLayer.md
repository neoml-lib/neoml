# CNotLayer Class

<!-- TOC -->

- [CNotLayer Class](#cnotlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that implements logical not operation over integer data. The formula is:

```c++
not(x) = x == 0 ? 1 : 0;
```

## Settings

The layer has no settings.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with integer data of any size.

## Outputs

The single output returns a blob of the same size where each element has been changed via the aforementioned formula.
