# CQrnnLayer Class

<!-- TOC -->

- [CQrnnLayer Class](#cqrnnlayer-class)
    - [Settings](#settings)
        - [Hidden layer size](#hidden-layer-size)
        - [Window size](#window-size)
        - [Window stride](#window-stride)
        - [Padding](#padding)
        - [Activation function](#activation-function)
        - [Dropout](#dropout)
        - [Reverse sequences](#reverse-sequences)
    - [Trainable parameters](#trainable-parameters)
        - [Filters](#filters)
        - [Free terms](#free-terms)
    - [Inputs](#inputs)
        - [First input size](#first-input-size)
        - [Second input size](#second-input-size)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a quasi-recurrent layer that can be applied to a set of vector sequences.

The output is a sequence of vectors, each of `GetHiddenSize()` size.

Unlike LSTM or GRU, this layer performs most of calculations before the recurrent part.
That leads to significant performance improvement on GPU.
It's achieved by using [time convolution](ConvolutionLayers/TimeConvLayer.md).

The realization of this layer is based on [this article](https://arxiv.org/abs/1611.01576).

## Settings

### Hidden layer size

```c++
void SetHiddenSize(int hiddenSize);
```

Sets the hidden layer size. It affects the output size.

### Window size

```c++
void SetWindowSize(int windowSize);
```

Sets the size of the window used in time convolution.

### Window stride

```c++
void SetStride(int stride);
```

Sets the stride of the window used in time convolution.

### Padding

```c++
void SetPaddingFront(int paddingFront);
```

Specifies how many zero elements should be added at the beginning of a sequence before performing convolution. The default value is `0`, that is, no padding used.

### Activation function

```c++
void SetActivation( TActivationFunction newActivation );
```

Sets the activation function that is used in `update` gate. By default, `AF_Tanh` is used.

### Dropout

```c++
void SetDropout(float rate);
```

Sets the dropout probability in `forget` gate.

### Reverse sequences

```c++
void SetReverseSequence( bool isReverseSequense )
```

Turns on processing sequences in reversed order.

## Trainable parameters

### Filters

```c++
CPtr<CDnnBlob> GetFilterData() cons;
```

The filters containing the weights for each gate. The filters are represented by a [blob](DnnBlob.md) of the following dimensions:

- `BatchLength` is equal to `1`
- `BatchWidth` is equal to `3 * GetHiddenSize()`
- `Height` is equal to `GetWindowSize()`
- `Width` is equal to `1`
- `Depth` is equal to `1`
- `Channels` is equal to the inputs `Height * Width * Depth * Channels`

The `BatchWidth` axis corresponds to the gate weights, in the following order:

```c++
G_Update, // update gate (Z in the article)
G_Forget, // forget gate (F in the article)
G_Output, // output gate (O in the article)
```

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const
```

The free terms are represented by a blob of the total size `3 * GetHiddenSize()`. The order in which they correspond to the gates is the same as [above](#filters).

## Inputs

The layer may have 1 to 2 inputs:

1. The set of vector sequences.
2. *[Optional]* The initial state of the recurrent part before the first step. If this input is not specified, the initial state is all zeros.

### First input size

- `BatchLength` - the length of one vector sequence.
- `BatchWidth` - the number of vector sequences in the input set.
- `Height * Width * Depth * Channels` - the size of each vector in the sequence.

### Second input size

- `BatchLength`, `Height`, `Width` and `Depth` should be `1`.
- `BatchWidth` must be equal to the `BatchWidth` of the first input.
- `Channels` must be equal to `GetHiddenSize()`.

## Outputs

The only output contains a blob with the results. The output blob dimensions are:

- `BatchLength` can be calculated from the input as `(BatchLength + GetPaddingFront() - (GetWindowSize() - 1)) / GetStride() + 1)`.
- `BatchWidth` is equal to the inputs' `BatchWidth`.
- `ListSize`, `Height`, `Width` and `Depth` are equal to `1`.
- `Channels` is equal to `GetHiddenSize()`.
