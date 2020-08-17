# CPositionalEmbeddingLayer Class

<!-- TOC -->

- [CPositionalEmbeddingLayer Class](#cpositionalembeddinglayer-class)
    - [Settings](#settings)
        - [Vector types](#vector-types)
    - [Trainable parameters](#trainable-parameters)
        - [Additive positional representations](#additive-positional-representations)
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
    // Non-learnable (used in transformers). https://arxiv.org/abs/1807.03819
    // Additional restrictions on input size: Depth == Height == Width == 1
    PET_Transformers,

    PET_EnumCount
};

void SetType( TPositionalEmbeddingType newType );
```

Vector types.

## Trainable parameters

### Additive positional representations

```c++
CPtr<CDnnBlob> GetAddends() const;
```

Vector representations of positions in a sequence, added to input vectors. Trained if `GetType()` is `PET_LearnableAddition` or `PET_LearnableMultAddition`.

Represented by a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength` and `BatchWidth` are equal to `1`
- other dimensions are equal to the corresponding ones of the input blob.

## Inputs

The single input accepts a blob that contains a batch of sequences of vectors, of the dimensions:

- `BatchLength` must be equal to `1`
- `BatchWidth` - amount of sequences in the batch
- `ListSize` - length of sequences
- `Height * Width * Depth * Channels` - vector size
  - if `GetType() == PET_Transformers` then `Height`, `Width` and `Depth` must be equal to `1`

## Outputs

There is only one output, which returns a blob of the same size as the input blob.

If `GetType() == PET_Transformers` then the result is equal to:

```c++
result[i][j][k] = input[i][j][k] + sin( j / pow( 10000, ( k / vectorSize ) ) )
```

where:

- `i` is the index of sequence in batch (from `0` to `BatchWidth - 1`)
- `j` is the position of vector in sequence (from `0` to `ListSize - 1`)
- `vectorSize` is the vector length (`Height * Width * Depth * Channels`)
- `k` is the index of the element in vector (from `0` to `vectorSize - 1`)

If `GetType() == PET_LearnableAddition` then the result is equal to:

```c++
result[i][j] = input[i][j] + GetAddend()[j]
```

where:

- `i` is the index of sequence in batch (from `0` to `BatchWidth - 1`)
- `j` is the position of vector in sequence (from `0` to `ListSize - 1`).
