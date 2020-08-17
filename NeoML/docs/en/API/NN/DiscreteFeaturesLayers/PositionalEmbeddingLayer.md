# CPositionalEmbeddingLayer Class

<!-- TOC -->

- [CPositionalEmbeddingLayer Class](#cpositionalembeddinglayer-class)
    - [Settings](#settings)
        - [Vector types](#vector-types)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that maps vectors with positions in sequence and trains these vectors in some cases.

## Settings

### Vector types

```c++
// Vector types
enum TPositionalEmbeddingType {
    // Learnable, addition-only Y = X + embedding
    PET_LearnableAddition = 0,
    // Learnable, linear transform Y = a * X + b
    PET_LearnableMultAddition,
    // Non-learnable (used in transformers). https://arxiv.org/abs/1807.03819
    // Additional restrictions on input size: Depth == Height == Width == 1
    PET_Transformers,

    PET_EnumCount
};

void SetType( TPositionalEmbeddingType newType );
```

Vector types.

## Trainable parameters

If `GetType() == PET_Transformers` then this layer doesn't have trainable parameters and the result is equal to:

```c++
result[i][j][k] = input[i][j][k] + sin( j / pow( 10000, ( k / vectorSize ) ) )
```

where:

- `i` is the index of sequence in batch (from `0` to `BatchWidth - 1`)
- `j` is the position of vector in sequence (from `0` to `ListSize - 1`)
- `vectorSize` is the vector length (`Height * Width * Depth * Channels`)
- `k` is the index of the element in vector (from `0` to `vectorSize - 1`)

If `GetType() == PET_LearnableAddition` then this layer trains one set of vectors (`B`) and the result is equal to:

```c++
result[i][j] = input[i][j] + B[j]
```

where:

- `i` is the index of sequence in batch (from `0` to `BatchWidth - 1`)
- `j` is the position of vector in sequence (from `0` to `ListSize - 1`).

If `GetType() == PET_LearnableMultAddition` then this layers trains two sets of vectors (`A` and `B`) and the result is equal to:

```c++
result[i][j] = ( input[i][j] * A[j] ) + B[j]
```

where:

- `i` is the index of sequence in batch (from `0` to `BatchWidth - 1`)
- `j` is the position of vector in sequence (from `0` to `ListSize - 1`)
- `*` multiplication of two vectors element by element.

## Inputs

The single input accepts a blob that contains a batch of sequences of vectors, of the dimensions:

- `BatchLength` must be equal to `1`
- `BatchWidth` - amount of sequences in the batch
- `ListSize` - length of sequences
- `Height * Width * Depth * Channels` - vector size
  - if `GetType() == PET_Transformers` then `Height`, `Width` and `Depth` must be equal to `1`

## Outputs

There is only one output, which returns a blob of the same size as the input blob.
