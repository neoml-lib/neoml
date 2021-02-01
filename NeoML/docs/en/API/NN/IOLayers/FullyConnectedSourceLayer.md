# CFullyConnectedSourceLayer Class

<!-- TOC -->

- [CFullyConnectedSourceLayer Class](#cfullyconnectedsourcelayer-class)
    - [Settings](#settings)
        - [Network input](#network-input)
        - [The number of vectors in one batch](#the-number-of-vectors-in-one-batch)
        - [The maximum number of batches in memory](#the-maximum-number-of-batches-in-memory)
        - [The value for empty spaces](#the-value-for-empty-spaces)
        - [The label data type](#the-label-data-type)
        - [The number of elements](#the-number-of-elements)
        - [Using the free terms](#using-the-free-terms)
    - [Trainable parameters](#trainable-parameters)
        - [Weight matrix](#weight-matrix)
        - [Free terms](#free-terms)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that can pass the data from an object implementing the [`IProblem`](../../ClassificationAndRegression/Problems.md) interface into the network, multiplying the `IProblem` vectors by a trainable weights matrix.

It is a more efficient implementation of the combination
of [CProblemSourceLayer](ProblemSourceLayer.md) and [CFullyConnectedLayer](../FullyConnectedLayer.md).

## Settings

### Network input

```c++
void SetProblem(const CPtr<const IProblem>& problem);
```

Sets the `IProblem` with the data that must be passed into the network.

### The number of vectors in one batch

```c++
void SetBatchSize(int batchSize);
```

Sets the number of vectors that are passed into the network from `GetProblem()` on one run.

On the first run, the first `GetBatchSize()` vectors are passed into the network, then the second `GetBatchSize()`, etc. After the last vector is passed, the first vector is passed again, and so on.

### The maximum number of batches in memory

```c++
void SetMaxBatchCount( int newMaxBatchCount );
```

Sets the upper limit to the number of batches stored in memory. The default value is `0`, which means that all data from `GetProblem()` is loaded into memory.

### The label data type

```c++
void SetLabelType( TDnnType newLabelType );
```

Sets the data type for the vectors' class labels.

### The number of elements

```c++
void SetNumberOfElements(int newNumberOfElements);
```

### Using the free terms

```c++
void SetZeroFreeTerm(bool _isZeroFreeTerm);
```

Specifies if the free terms should be used. If you set this value to `true`, the free terms vector will be set to all zeros and won't be trained. By default, this value is set to `false`.

## Trainable parameters

### Weight matrix

```c++
CPtr<CDnnBlob> GetWeightsData() const;
```

The weight matrix is a [blob](../DnnBlob.md) of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to `GetNumberOfElements()`
- `Height`, `Width`, and `Depth` are equal to `1`
- `Channels` is equal to the vector length for `IProblem`

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const;
```

The free terms are represented by a blob of the total size equal to `GetNumberOfElements()`.

## Inputs

The layer has no inputs.

## Outputs

The layer has three outputs.

The first output contains a blob with data vectors from `IProblem`, of the dimensions:

- `BatchWidth` is equal to `GetBatchSize()`
- `Chahhels` is equal to `GetNumberOfElements()`
- the other dimensions are equal to `1`

The second output contains a blob with correct class labels for the vectors from `IProblem`. The data is of the `GetLabelType()` type. The blob dimensions are:

- `BatchWidth` is equal to `GetBatchSize()`
- `Channels` is equal to `1` for `int` data type and to the number of classes in `IProblem` otherwise
- the other dimensions are equal to `1`

The third output contains the vector weights from `IProblem`. The blob dimensions are:

- `BatchWidth` is equal to `GetBatchSize()`
- the other dimensions are equal to `1`
