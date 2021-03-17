# CConcatBatchWidthLayer Class

<!-- TOC -->

- [CConcatBatchWidthLayer Class](#cconcatbatchwidthlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that concatenates the input blobs along the `BatchWidth` dimension.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer accepts an arbitrary number of inputs, each containing a [blob](../DnnBlob.md) with data:

- `BatchLength`, `ListSize`, `Height`, `Width`, `Depth`, and `Channels` dimensions must be the same for all inputs. 
- The `BatchWidth` dimension may vary.

## Outputs

The layer has one output which contains a blob with the result of concatenation. The dimensions of the blob are:

- `BatchLength`, `ListSize`, `Height`, `Width`, `Depth`, and `Channels` are equal to the corresponding inputs' dimensions.
- `BatchWidth` is equal to the sum of `BatchWidth` over all the inputs.
