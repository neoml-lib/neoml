# CQrnnLayer Class

<!-- TOC -->

- [CQrnnLayer Class](#cqrnnlayer-class)
    - [Settings](#settings)
        - [Pooling type](#pooling-type)
        - [Hidden layer size](#hidden-layer-size)
        - [Window size](#window-size)
        - [Window stride](#window-stride)
        - [Padding](#padding)
        - [Activation function](#activation-function)
        - [Dropout](#dropout)
        - [Recurrent mode](#recurrent-mode)
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

Unlike LSTM or GRU, this layer performs most of calculations before the recurrent part, which leads to significant performance improvement on GPU.
It's achieved by using [time convolution](ConvolutionLayers/TimeConvLayer.md).

Based on [this article](https://arxiv.org/abs/1611.01576).

## Settings

### Pooling type

```c++
// Different poolings used in QRNN
enum TPoolingType {
    PT_FPooling, // f-pooling from article, uses 2 gates (Update, Forget)
    PT_FoPooling, // fo-pooling from article, uses 3 gates (Update, Forget, Output)
    PT_IfoPooling, // ifo pooling from article, uses 4 gates (Update, Forget, Output, Input)

    PT_Count
};

void SetPoolingType(TPoolingType newPoolingType);
```

Sets the pooling type. Pooling is the recurrent part of the QRNN layer.
The exact formulas are given in [the article](https://arxiv.org/abs/1611.01576).

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
void SetPaddingFront(int padding);
```

Specifies how many zero elements should be added to the begginnings of the sequences before performing convolution. The default value is `0`, that is, no padding used.

```c++
void SetPaddingBack(int padding);
```

Specifies how many zero elements should be added to the ends of the sequences before performing convolution. The default value is `0`, that is, no padding used.

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

### Recurrent mode

```c++
// Different approaches in sequence processing
enum TRecurrentMode {
    RM_Direct,
    RM_Reverse,

    // Bidirectional mode where two recurrent parts share the same time convolution
    RM_BidirectionalConcat, // returns the concatenation of direct and reverse recurrents
    RM_BidirectionalSum, // returns the sum of direct and reverse recurrents
    // If you want to use bidirectional qrnn with two separate time convolutions create 2 CQrnnLayers
    // and merge the results by CConcatChannelsLayer or CEltwiseSumLayer

    RM_Count
};

void SetRecurrentMode( TRecurrentMode newMode );
```

Sets the way this layer processes input sequences.

## Trainable parameters

### Filters

```c++
CPtr<CDnnBlob> GetFilterData() cons;
```

The filters containing the weights for each gate. The filters are represented by a [blob](DnnBlob.md) of the following dimensions:

- `BatchLength` is equal to `1`
- `BatchWidth` is equal to `gates * GetHiddenSize()`, where `gates` is `2` if `PT_FPooling` is used, `3` in case of `PT_FoPooling` and `4` in case of `PT_IfoPooling`
- `Height` is equal to `GetWindowSize()`
- `Width` is equal to `1`
- `Depth` is equal to `1`
- `Channels` is equal to the inputs `Height * Width * Depth * Channels`

The `BatchWidth` axis corresponds to the gate weights, in the following order:

```c++
G_Update, // update gate (Z in the article)
G_Forget, // forget gate (F in the article)
G_Output, // output gate if used (O in the article)
G_Input, // input gate if used (I in the article)
```

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const
```

The free terms are represented by a blob of the total size of `BatchWidth` of filters above. The order in which they correspond to the gates is the same as [above](#filters).

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

- `BatchLength` can be calculated from the input as `(BatchLength + GetPaddingFront() + GetPaddingBack() - (GetWindowSize() - 1)) / GetStride() + 1)`.
- `BatchWidth` is equal to the inputs' `BatchWidth`.
- `ListSize`, `Height`, `Width` and `Depth` are equal to `1`.
- `Channels` is equal to `2 * GetHiddenSize()` if `GetRecurrentMode()` is `RM_BidirectionalConcat`. Otherwise it's equal to `GetHiddenSize()`.
