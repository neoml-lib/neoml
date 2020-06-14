# CArgmaxLayer Class

<!-- TOC -->

- [CArgmaxLayer Class](#cargmaxlayer-class)
    - [Settings](#settings)
        - [Dimension](#dimension)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that finds, for a single input, the coordinates of the maximum elements along a given dimension.

## Settings

### Dimension

```c++
void SetDimension(TBlobDim d);
```

Sets the dimension along which the maximums should be found.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The single input accepts a blob of any size.

## Outputs

The single output contains a blob with `int` data of the dimensions:

- the `GetDimension()` dimension equals `1`
- the other dimensions are the same as in the input blob

The elements of the blob contain the coordinates of the maximums taken along the specified dimension.
