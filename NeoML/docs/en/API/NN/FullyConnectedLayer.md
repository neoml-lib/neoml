# CFullyConnectedLayer Class

<!-- TOC -->

- [CFullyConnectedLayer Class](#cfullyconnectedlayer-class)
    - [Settings](#settings)
        - [Output vector length](#output-vector-length)
        - [Using the free terms](#using-the-free-terms)
    - [Trainable parameters](#trainable-parameters)
        - [Weight matrix](#weight-matrix)
        - [Free terms](#free-terms)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a fully connected layer. The layer multiplies each of the input vectors by the weight matrix and adds the free term vector to the result.

## Settings

### Output vector length

```c++
void SetNumberOfElements(int newNumberOfElements);
```

Sets the length of each vector in the output.

### Using the free terms

```c++
void SetZeroFreeTerm(bool isZeroFreeTerm);
```

Specifies if the free terms should be used. If you set this value to `true`, the free terms vector will be set to all zeros and won't be trained. By default, this value is set to `false`.

## Trainable parameters

### Weight matrix

```c++
CPtr<CDnnBlob> GetWeightsData() const;
```

The weight matrix is a [blob](../DnnBlob.md) of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to `GetNumberOfElements()`
- `Height`, `Width`, `Depth`, and `Channels` are equal to the same dimensions of the first input

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const;
```

The free terms are represented by a blob of the total size equal to `GetNumberOfElements()`.

## Inputs

Each input accepts a blob with a set of vectors, of the dimensions:

- `BatchLength * BatchWidth * ListSize` is the number of vectors in the set.
- `Height * Width * Depth * Channels` is each vector length. It should be the same for all inputs.

## Outputs

For each input, the corresponding output contains a blob with the result, of the dimensions:

- `BatchLength` is equal to the input `BatchLength`
- `BatchWidth` is equal to the input `BatchWidth`
- `ListSize` is equal to the input `ListSize`
- `Height`, `Width`, and `Depth` are equal to `1`
- `Channels` is equal to `GetNumberOfElements()`
