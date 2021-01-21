# CDropoutLayer Class

<!-- TOC -->

- [CDropoutLayer Class](#cdropoutlayer-class)
    - [Settings](#settings)
        - [Dropout rate](#dropout-rate)
        - [Spatial dropout mode](#spatial-dropout-mode)
        - [Batchwise dropout mode](#batchwise-dropout-mode)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that randomly sets some elements of a single input to `0`.

If the blob `BatchLength` is greater than `1`, all elements along the same `BatchLength` coordinate will use the same mask.

When the network is not being trained (for example, you are doing a test run), the dropout will not happen.

## Settings

### Dropout rate

```c++
void SetDropoutRate( float value );
```

Sets the proportion of elements that will be set to `0`.

### Spatial dropout mode

```c++
void SetSpatial( bool value );
```

Turns on and off the `spatial` dropout mode. When this mode is on, the whole contents of a channel will be filled with zeros, instead of elements one by one. It may be useful for convolutional networks.

By default, spatial mode is off.

### Batchwise dropout mode

```c++
void SetBatchwise( bool value );
```

Turns on and off the `batchwise` dropout mode. When this mode is on, the same mask will be used along the same `BatchWidth` coordinate. The mode may be useful when the input size is large.

By default, batchwise mode is off.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a data blob of arbitrary size.

## Outputs

The single output returns a blob of the same size with some of the elements set to `0`. Note that this will happen only during training; when you are running the network without training no elements are dropped out.
