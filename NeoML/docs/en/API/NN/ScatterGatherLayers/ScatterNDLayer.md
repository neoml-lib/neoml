# CScatterNDLayer Class

<!-- TOC -->

- [CScatterNDLayer Class](#cscatterndlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that scatters updates over the given data blob.

```c++
for( int updateIndex = 0; updateIndex < UpdateCount; ++updateIndex )
    data[indices[updateIndex]] = updates[updateIndex];
```

where `indices[...]` is an integer vector of `IndexDims` elements which contains coordinates in the first `IndexDims` dimensions of the data blob.

## Settings

There are no settings for this layer.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

Layer has 3 inputs:

1. The set of objects of any data type. The product of first `IndexDims` dimensions is `ObjectCount` and the product of the rest of dimensions is `ObjectSize`.
2. The set of indices of integer data type of the following shape:
- `BD_Channels` of the blob must be equal to the `IndexDims`
- The product of the other dimensions must be equal to the `UpdateCount`.
3. The set of updates of the same data type as first input. It must contain `UpdateCount * ObjectSize` elements regardless of shape.

## Outputs

The only output contains updated data of the same size and type as the first input.
