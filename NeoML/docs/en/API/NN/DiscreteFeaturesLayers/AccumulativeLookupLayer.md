# CAccumulativeLookupLayer Class

<!-- TOC -->

- [CAccumulativeLookupLayer Class](#caccumulativelookuplayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
        - [The representation table](#the-representation-table)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that trains fixed-length vector representations for the values of a discrete feature.

See [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [GloVe](https://en.wikipedia.org/wiki/GloVe_(machine_learning)), etc.

This layer can work only with one feature; when several values of the feature are passed, the sum of the corresponding vectors is returned.

## Settings

```c++
// Size of the representation table
struct CLookupDimension {
    int VectorCount; // the number of vectors
    int VectorSize; // the vector length
};

void SetDimension( const CLookupDimension& newDimension );
```

Sets the size of the vector table.

## Trainable parameters

### The representation table

```c++
CPtr<CDnnBlob> GetEmbeddings() const;
```

Gets the table with the trained vectors. The blob storing the table has the following dimensions:

- `BatchLength` is equal to `GetDimension().VectorCount`;
- `BatchWidth` is equal to `GetDimension().VectorSize`.

## Inputs

The single input accepts a blob with `int` data that contains the feature values, of the dimensions:

- `BatchLength * BatchWidth * ListSize` is the number of different values the feature can take
- `Height * Width * Depth * Channels` is the number of values in the set

## Outputs

The single output contains a blob with the sum of vector representations of the given feature values. The blob dimensions are:

- `BatchLength` is equal to the input `BatchLength`
- `BatchWidth` is equal to the input `BatchWidth`
- `ListSize` is equal to the input `ListSize`
- `Height` is equal to `1`
- `Width` is equal to `1`
- `Depth` is equal to `1`
- `Channels` is equal to `GetDimension().VectorSize`.
