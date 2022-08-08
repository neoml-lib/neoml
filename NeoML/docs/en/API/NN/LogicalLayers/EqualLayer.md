# CEqualLayer Class

<!-- TOC -->

- [CEqualLayer Class](#cequallayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that implements elentwise comparison between
2 inputs

```c++
equal[i] = first_input[i] == second_input[i] ? 1 : 0;
```

## Settings

The layer has no settings.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

Layer accepts any pair of blobs of the same size and data type.

## Outputs

The single output returns an integer blob of the same size where each element contains the result of the comparison of 2 input blobs.

