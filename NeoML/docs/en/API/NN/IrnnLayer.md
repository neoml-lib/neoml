# CIrnnLayer Class

<!-- TOC -->

- [CIrnnLayer Class](#cirnnlayer-class)
    - [Settings](#settings)
        - [Hidden layer size](#hidden-layer-size)
        - [Identity scale](#identity-scale)
        - [Input weight std](#input-weight-std)
    - [Trainable parameters](#trainable-parameters)
        - [Input weight matrix](#input-weight-matrix)
        - [Input free terms](#input-free-terms)
        - [Recurrent weight matrix](#recurrent-weight-matrix)
        - [Recurrent free terms](#recurrent-free-terms)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements IRNN from this [article](https://arxiv.org/pdf/1504.00941.pdf).

It's a simple recurrent unit with the following formula:

```c++
    Y_t = ReLU( FC_input( X_t ) + FC_recur( Y_t-1 ) )
```

where `FC_*` are fully-connected layers.

The crucial point of this layer is weights initialization.
The weight matrix of `FC_input` is initialized from `N(0, inputWeightStd)` where `inputWeightStd` is a layer setting.
The weight matrix of `FC_recur` is initialized with an identity matrix multiplied by `identityScale` setting.

## Settings

### Hidden layer size

```c++
void SetHiddenSize( int size );
```

Sets the hidden layer size. It affects the output size.

### Identity scale

```c++
void SetIdentityScale( float scale );
```

Sets the scale of identity matrix, used for the initialization of recurrent weights.

### Input weight std

```c++
void SetInputWeightStd( float var );
```

Sets the standard deviation for input weights.

## Trainable parameters

### Input weight matrix

```c++
CPtr<CDnnBlob> GetInputWeightsData() const;
```

The weight matrix of the `FC_input` from formula.

It has the following shape:

- `BatchLength * BatchWidth * ListSize` is equal to `GetHiddenSize()`
- `Height * Width * Depth * Channels` is equal to the product of the same dimensions of the input.

### Input free terms

```c++
CPtr<CDnnBlob> GetInputFreeTermData() const
```

The free terms of the `FC_input`. It's represented by a blob of the total size `GetHiddenSize()`.

### Recurrent weight matrix

```c++
CPtr<CDnnBlob> GetRecurWeigthsData() const;
```

The weight matrix of the `FC_recur` from formula.

It has the following shape:

- `BatchLength * BatchWidth * ListSize` is equal to `GetHiddenSize()`
- `Height * Width * Depth * Channels` is equal to `GetHiddenSize()`

### Recurrent free terms

```c++
CPtr<CDnnBlob> GetRecurrentFreeTermData() const
```

The free terms of the `FC_recur`. It's represented by a blob of the total size `GetHiddenSize()`.

## Inputs

The single input of this layer accepts the set of vector sequences of the following shape:

- `BatchLength` - the length of one vector sequence.
- `BatchWidth * ListSize` - the number of vector sequences in the input set.
- `Height * Width * Depth * Channels` - the size of each vector in the sequence.

## Outputs

The single output returns a blob of the following size:

- `BatchLength`, `BatchWidth`, and `ListSize` are equal to the same sizes of the first input.
- `Height`, `Width`, and `Depth` equal `1`.
- `Channels` equals `GetHiddenSize()`.
