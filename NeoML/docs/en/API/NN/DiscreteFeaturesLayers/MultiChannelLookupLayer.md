# CMultichannelLookupLayer Class

<!-- TOC -->

- [CMultichannelLookupLayer Class](#cmultichannellookuplayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
        - [The representation table](#the-representation-table)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that trains fixed-length vector representations for the values of several discrete features.

See [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [GloVe](https://en.wikipedia.org/wiki/GloVe_(machine_learning)), etc.

The layer can work with several representation tables at once, one per each feature.

## Settings

```c++
// Size of a representation table
struct CLookupDimension {
    int VectorCount; // the number of vectors
    int VectorSize; // the vector length
};

void SetDimensions(const CArray<CLookupDimension>&);
```

Sets the array of vector table sizes. The array length specifies the number of tables, and each of its elements specifies the number and length of vectors in the corresponding table.

## Trainable parameters

```c++
const CDnnBlob& GetEmbeddings(int i) const;
```

This layer trains several vector tables. Each `i`th table is represented by a [blob](DnnBlob.md) of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to `GetDimensions()[i].VectorCount`
- `Height * Width * Depth * Channels` is equal to `GetDimensions()[i].VectorSize`

## Inputs

The layer supports multiple inputs.
Each input accepts a blob with `float` or `int` data that contains the feature values, of the dimensions:

- `BatchLength * BatchWidth * ListSize * Height * Width * Depth` is equal to the number of features in the set.
- `Channels` is the dimension along which the feature values for different sets are stored; so it should not be smaller than the number of feature sets.

## Outputs

The number of outputs is equal to the number of inputs.
Each output contains a blob with the results, of the dimensions:

- `BatchLength`, `BatchWidth`, `ListSize`, `Height`, `Width`, `Depth` are equal to these dimensions for the corresponding input
- `Channels` is equal to the sum of the vector lengths of all sets and the additional channels (if the corresponding input `Channels` is more than the number of tables).
