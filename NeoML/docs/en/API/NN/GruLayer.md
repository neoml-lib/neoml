# CGruLayer Class

<!-- TOC -->

- [CGruLayer Class](#cgrulayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
        - [Gate weights matrix](#gate-weights-matrix)
        - [Gate free terms](#gate-free-terms)
        - [Output weights matrix](#output-weights-matrix)
        - [Output free terms](#output-free-terms)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) layer that works with a set of vector sequences.

The result is a vector sequence of the same length, with each vector the length of `GetHiddenSize()`.

## Settings

## Trainable parameters

### Gate weights matrix

```c++
CPtr<CDnnBlob> GetGateWeightsData() const;
```

The gate weights are put into a two-dimensional matrix represented by a [blob](DnnBlob.md) of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to `2 * GetHiddenSize()`
- `Height * Width * Depth * Channels` is equal to the same dimension of the input plus `GetHiddenSize()`.

Along the `BatchLength * BatchWidth * ListSize` axis the gate weights are sorted in the following order:

```c++
G_Update = 0, // Update gate
G_Reset,      // Reset gate
```

Along the `Height * Width * Depth * Channels` axis the weights are put into following order:

- from the start to the input `Height * Width * Depth * Channels` the weights for the input vectors
- the rest `GetHiddenSize()` coordinates correspond to the weights for the previous step result.

### Gate free terms

```c++
CPtr<CDnnBlob> GetGateFreeTermData() const;
```

The free terms for the gates are represented by a blob of `2 * GetHiddenSize()` total size.

### Output weights matrix

```c++
CPtr<CDnnBlob> GetMainWeightsData() const;
```

The output weights are put into a two-dimensional matrix represented by a blob of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to `GetHiddenSize()`
- `Height * Width * Depth * Channels` is equal to the same dimension fo the input plus `GetHiddenSize()`

### Output free terms

The free terms for the output are represented by a blob of `GetHiddenSize()` total size.

## Inputs

The layer has 1 to 2 inputs:

1. The set of vector sequences.
2. *[Optional]* The initial previous step result that should be used on the first step. If you do not connect this input all zeros will be used.

## Output

The single output returns a blob of the dimensions:

- `BatchLength`, `BatchWidth`, and `ListSize` equal the corresponding input dimensions
- `Height`, `Width`, and `Depth` are equal to `1`
- `Channels` equals `GetHiddenSize()`
