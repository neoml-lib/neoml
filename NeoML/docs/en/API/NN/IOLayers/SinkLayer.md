# CSinkLayer Class

<!-- TOC -->

- [CSinkLayer Class](#csinklayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Retrieving the data blob](#retrieving-the-data-blob)

<!-- /TOC -->

This class implements a sink layer that serves to pass a blob of data out of the neural network.

## Settings

There are no settings for this layer.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

The single input accepts a blob of arbitrary size, containing any kind of data.

## Outputs

The layer has no outputs.

### Retrieving the data blob

To retrieve the blob with the result of the last neural network run, use the `GetBlob()` method.

```c++
const CPtr<CDnnBlob>& GetBlob() const;
```
