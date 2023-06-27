# CLstmLayer Class

<!-- TOC -->

- [CLstmLayer Class](#clstmlayer-class)
    - [Settings](#settings)
        - [Hidden layer size](#hidden-layer-size)
        - [Variational dropout](#variational-dropout)
        - [Activation function](#activation-function)
    - [Trainable parameters](#trainable-parameters)
        - [Weight matrix](#weight-matrix)
        - [Free terms](#free-terms)
    - [Inputs](#inputs)
        - [First input size](#first-input-size)
        - [Other inputs size](#other-inputs-size)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a long short-term memory ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)) layer that can be applied to a set of vector sequences.

The output is a sequence containing the same number of vectors, each of `GetHiddenSize()` size.

## Settings

### Hidden layer size

```c++
void SetHiddenSize(int size);
```

Sets the hidden layer size. It affects the output size and the size of the *state vector* inside the LSTM.


### Variational dropout

```c++
void SetDropoutRate(float newDropoutRate);
```

Sets the dropout probability. If this value is set, the operation will be performed on the input combined with the output of the last run; the result will be passed to the fully connected layer.

### Activation function

```c++
void SetRecurrentActivation( TActivationFunction newActivation );
```

Sets the activation function that is used in `forget`, `reset`, and `input` gates. By default, `AF_Sigmoid` is used.

## Trainable parameters

### Weight matrix

```c++
CPtr<CDnnBlob> GetWeightsData() const;
```

The weight matrix containing the weights for each gate. The matrix is represented by a [blob](DnnBlob.md) of the following dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to `4 * GetHiddenSize()`.
- `Height * Width * Depth * Channels` is equal to the sum of the same dimension of the input and `GetHiddenSize()`.

The `BatchLength * BatchWidth * ListSize` axis corresponds to the gate weights, in the following order:

```c++
G_Main = 0, // The main output data
G_Forget,   // Forget gate
G_Input,    // Input gate
G_Reset,    // Reset gate
```

The `Height * Width * Depth * Channels` axis corresponds to the weights:

- `0` to the input size: weights that serve as coefficients for the vectors of the input sequence;
- the rest of the coordinates (up to `HiddenSize`) correspond to the weights that serve as coefficients for the output of the previous step.

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const
```

The free terms are represented by a blob of the total size `4 * GetHiddenSize()`. The order in which they correspond to the gates is the same as [above](#weight-matrix).

## Inputs

The layer may have 1 to 3 inputs:

1. The set of vector sequences.
2. *[Optional]* The initial state of the LSTM layer before the first step. If this input is not specified, the initial state is all zeros.
3. *[Optional]* The initial value of the "previous output" to be used on the first step. If this input is not specified, all zeros are used.

### First input size

- `BatchLength` - the length of one vector sequence.
- `BatchWidth` - the number of vector sequences in the input set.
- `ListSize` should be `1`.
- `Height * Width * Depth * Channels` - the size of each vector in the sequence.

### Other inputs size

- `BatchLength` and `ListSize` should be `1`.
- `BatchWidth` should be equal to the `BatchWidth` of the first input.
- `Height * Width * Depth * Channels` must be equal to the `GetHiddenSize()`.

## Outputs

The layer has two outputs:

1. The result of the current step.
2. The layer history.

Both outputs are of the following size:

- `BatchLength` and `BatchWidth` are equal to the same sizes of the first input.
- `ListSize`, `Height`, `Width`, and `Depth` equal `1`.
- `Channels` equals `GetHiddenSize()`.