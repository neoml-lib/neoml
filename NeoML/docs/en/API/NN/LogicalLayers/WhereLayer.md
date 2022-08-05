# CWhereLayer Class

<!-- TOC -->

- [CWhereLayer Class](#cwherelayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that implements elentwise merge of blobs from the second and third input
based on the values from the first input

```c++
where[i] = first_input[i] != 0 ? second_input[i] : third_input[i];
```

## Settings

The layer has no settings.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer has 3 inputs:

1. Integer blob of any size with the mask
2. Blob of the same size as first of any data type with the values used when the mask is not zero.
3. Blob of the same size and data type as second with the values used when mask the mask is zero.

## Outputs

The single output returns a blob of the same size and data type as second input.
