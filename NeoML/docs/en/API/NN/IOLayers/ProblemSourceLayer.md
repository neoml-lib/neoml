# CProblemSourceLayer Class

<!-- TOC -->

- [CProblemSourceLayer Class](#cproblemsourcelayer-class)
    - [Settings](#settings)
        - [Network input](#network-input)
        - [The number of vectors in one batch](#the-number-of-vectors-in-one-batch)
        - [The value for empty spaces](#the-value-for-empty-spaces)
        - [The label data type](#the-label-data-type)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that can pass the data from an object implementing the [`IProblem`](../../ClassificationAndRegression/Problems.md) interface into the network.

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

### The value for empty spaces

```c++
void SetEmptyFill(float emptyFill);
```

Sets the value with which the empty spaces in data are filled. It is needed because the `IProblem` stores the vectors in sparse format.
The default value is `0`.

### The label data type

```c++
void SetLabelType( TDnnType newLabelType );
```

Sets the data type for the vectors' class labels.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

The layer has no inputs.

## Outputs

The layer has three outputs.

The first output contains a blob with data vectors from `IProblem`, of the dimensions:

- `BatchWidth` is equal to `GetBatchSize()`
- `Chahhels` is equal to the vector length in `GetProblem()`
- the other dimensions are equal to `1`

The second output contains a blob with correct class labels for the vectors from `IProblem`. The data is of the `GetLabelType()` type. The blob dimensions are:

- `BatchWidth` is equal to `GetBatchSize()`
- `Channels` is equal to `1` for `int` data type and to the number of classes in `IProblem` otherwise
- the other dimensions are equal to `1`

The third output contains the vector weights from `IProblem`. The blob dimensions are:

- `BatchWidth` is equal to `GetBatchSize()`
- the other dimensions are equal to `1`
