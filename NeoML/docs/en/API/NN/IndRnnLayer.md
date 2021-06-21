# CIndRnnLayer Class

<!-- TOC -->

- [CIndRnnLayer Class](#cindrnnlayer-class)
    - [Settings](#settings)
        - [Hidden layer size](#hidden-layer-size)
        - [Dropout rate](#dropout-rate)
        - [Reverse sequences](#reverse-sequences)
    - [Trainable parameters](#trainable-parameters)
        - [Weight matrix W](#weight-matrix-w)
        - [Weight vector U](#weight-vector-u)
        - [Free terms B](#free-term-b)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements IndRNN from this [article](https://arxiv.org/pdf/1803.04831.pdf).

It's a simple recurrent unit with the following formula:

```c++
    Y_t = sigmoid( W * X_t + B + U * Y_t-1 )
```

where:

- `W` and `B` are weight matrix and free terms of the fully-connected layer respectively (`W * X_t` means matrix-by-vector multiplication)
- `U` is a recurrent weights vector (`U * Y_t-1` means element-wise multiplication of 2 vectors of same length)

## Settings

### Hidden layer size

```c++
void SetHiddenSize( int size );
```

Sets the hidden layer size. It affects the output size.

### Dropout rate

```c++
void SetDropoutRate( float dropoutRate );
```

Sets the rate of dropout, applied to both input (`X_t`) and recurrent part (`Y_t-1`).

### Reverse sequences

```c++
void SetReverseSequence( bool reverse );
```

Elements of the sequences are processed in reversed order if this flag is set.

## Trainable parameters

### Weight matrix W

```c++
CPtr<CDnnBlob> GetInputWeights() const;
```

The weight matrix `W` from the formula.

It has the following shape:

- `BatchLength * BatchWidth * ListSize` is equal to `GetHiddenSize()`
- `Height * Width * Depth * Channels` is equal to the product of the same dimensions of the input.

### Weight vector U

```c++
CPtr<CDnnBlob> GetRecurrentWeights() const;
```

The weight vector `U` from the formula. It's represented by a blob of the total size `GetHiddenSize()`.

### Free terms B

```c++
CPtr<CDnnBlob> GetBias() const
```

The free terms `B` from the formula. It's represented by a blob of the total size `GetHiddenSize()`.

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
