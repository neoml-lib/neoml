# CDotProductLayer Class

<!-- TOC -->

- [CDotProductLayer Class](#cdotproductlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates the dot product of its two inputs. Each object in the first input is multiplied by the object with the corresponding index in the second input.

## Settings

The layer has no settings.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer has two inputs, which must contain blobs of the same dimensions.

## Outputs

The single output returns a blob with the results, of the dimensions:

- `BatchLength`, `BatchWidth`, `ListSize` are equal to these dimensions of the inputs;
- `Height`, `Width`, `Depth`, `Channels` are equal to `1`.
