# CSourceLayer Class

<!-- TOC -->

- [CSourceLayer Class](#csourcelayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that serves to input a blob of user data into the neural network.

## Settings

```c++
void SetBlob( CDnnBlob* blob );
```

Sets the blob with source data. It may be of arbitrary size and contain any kind of data.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

The layer has no inputs.

## Outputs

The single output contains the data blob that was passed into the last call of `SetBlob()`.
