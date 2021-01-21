# CConcatHeightLayer Class

<!-- TOC -->

- [CConcatHeightLayer Class](#cconcatheightlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that concatenates the input blobs along the `Height` dimension.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer accepts an arbitrary number of inputs, each containing a [blob](..\DnnBlob.md) with data:

- `BatchLength`, `BatchWidth`, `ListSize`, `Width`, `Depth`, and `Channels` dimensions must be the same for all inputs. 
- The `Height` dimension may vary.

## Outputs

The layer has one output which contains a blob with the result of concatenation. The dimensions of the blob are:

- `BatchLength`, `BatchWidth`, `ListSize`, `Width`, `Depth`, and `Channels` are equal to the corresponding inputs' dimensions.
- `Height` is equal to the sum of `Height` over all the inputs.
