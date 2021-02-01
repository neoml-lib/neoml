# CSoftmaxLayer Class

<!-- TOC -->

- [CSoftmaxLayer Class](#csoftmaxlayer-class)
    - [Settings](#settings)
        - [Normalization area](#normalization-area)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that calculates the `softmax` function on a vector set.

The following formula is applied to each of the vectors:

```c++
softmax(x[0], ... , x[n-1])[i] = exp(x[i]) / (exp(x[0]) + ... + exp(x[n-1]))
```

## Settings

### Normalization area

```c++
// The dimensions over which the vectors should be normalized
enum TNormalizationArea {
    NA_ObjectSize = 0,
    NA_BatchLength,
    NA_ListSize,
    NA_Channel,

    NA_Count
};

void SetNormalizationArea( TNormalizationArea newArea )
```

Specifies which dimensions of the input [blob](DnnBlob.md) constitute the vector length:

- `NA_ObjectSize` - *[Default]* the input blob will be considered to contain `BatchLength * BatchWidth * ListSize` vectors, each of `Height * Width * Depth * Channels` length.
- `NA_BatchLength` - the input blob will be considered to contain `BatchWidth * ListSize * Height * Width * Depth * Channels` vectors, each of `BatchLength` length.
- `NA_ListSize` - the input blob will be considered to contain `BatchLength * BatchWidth * Height * Width * Depth * Channels` vectors, each of `ListSize` length.
- `NA_Channel` - the input blob will be considered to contain `BatchLength * BatchWidth * ListSize * Height * Width * Depth` vectors, each of `Channels`

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a data blob of any size. The  [`GetNormalizationArea()`](normalization-area) setting determines which dimensions will be considered to consitute vector length.

## Outputs

The single output contains a blob of the same size with the result of `softmax` function applied to each vector.
