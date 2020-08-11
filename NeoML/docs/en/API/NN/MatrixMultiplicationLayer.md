# CMatrixMultiplicationLayer Class

<!-- TOC -->

- [CMatrixMultiplicationLayer Class](#cmatrixmultiplicationlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that multiplies matrices from two sets.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

This layer has 2 inputs. Each of them accepts the set of matrices:

- `BatchLength * BatchWidth * ListSize` - amount of matrices in sets, must be equal between inputs
- `Height * Width * Depth` - amount of rows
- `Channels` - amount of columns
- `Channels` of the first input must be equal to `Height * Width * Depth` of the second input

## Outputs

Single output returns the blob with multiplication results. The size of the blob:

- `BatchLength`, `BatchWidth`, `ListSize`, `Height`, `Width`, `Depth` are equal to these dimensions of the first input
- `Channels` is equal to the `Channels` of the second input.
