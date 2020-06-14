# CConcatObjectLayer Class

<!-- TOC -->

- [CConcatObjectLayer Class](#cconcatobjectlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that concatenates the input objects along the `Height`, `Width`, `Depth`, and `Channels` dimensions.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer accepts an arbitrary number of inputs, each containing a [blob](..\DnnBlob.md) with data:

- `BatchLength`, `BatchWidth`, and `ListSize` dimensions must be the same for all inputs. 
- other dimensions may vary.

## Outputs

The layer has one output which contains a blob with the result of concatenation. The dimensions of the blob are:

- `BatchLength`, `BatchWidth`, and `ListSize` are equal to the corresponding inputs' dimensions.
- `Height`, `Width`, and `Depth` are equal to `1`.
- `Channels` is equal to the sum of `Height * Width * Depth * Channels` over all the inputs.
