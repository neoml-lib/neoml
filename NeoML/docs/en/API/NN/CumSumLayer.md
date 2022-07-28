# CCumSumLayer Class

<!-- TOC -->

- [CCumSumLayer Class](#ccumsumlayer-class)
    - [Settings](#settings)
        - [Dimension](#dimension)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that performs cumulative sum along the given dimension.

## Settings

### Dimension

```c++
void SetDimension(TBlobDim newDim);
```

Sets the dimension along which the cumulative sum should be calculated.

### Reverse

```c++
void SetReverse(bool newReverse);
```

If set to `true` then cumulative sum will be calculated in reverse order.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The single input accepts a blob of any size and data type.

## Outputs

The single output contains a blob of the same size and data type.