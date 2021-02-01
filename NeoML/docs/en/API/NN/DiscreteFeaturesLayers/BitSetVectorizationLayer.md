# CBitSetVectorizationLayer Class

<!-- TOC -->

- [CBitSetVectorizationLayer Class](#cbitsetvectorizationlayer-class)
    - [Settings](#settings)
        - [Bitset size](#bitset-size)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that converts a bitset into vectors of ones and zeros.

## Settings

### Bitset size

```c++
void SetBitSetSize( int bitSetSize );
```

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with `int` data, of the dimensions:

- `BatchLength * BatchWidth * ListSize * Height * Width * Depth` is the number of bitsets
- `Channels` contains the bitset itself

## Outputs

The single output contains a blob of the dimensions:

- `Channels` is equal to `GetBitSetSize()`
- the other dimensions are equal to the input dimensions
