# CMatrixMultiplicationLayer Class

<!-- TOC -->

- [CMatrixMultiplicationLayer Class](#cmatrixmultiplicationlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that performs matrix multiplication operation for its two inputs.

It treats the first three dimensions (`BatchLength * BatchWidth * ListSize`) as the number of matrices, the next three (`Height * Width * Depth`) as the height of each and the last one (`Channels`) as the width of each matrix. So this layer implements a bunch of independent matrix multiplications.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

This layer has 2 inputs. Each of them accepts the set of matrices:

- `BatchLength * BatchWidth * ListSize` - the number of matrices in sets, must be equal between inputs
- `Height * Width * Depth` - matrix height
- `Channels` - matrix width; `Channels` of the first input must be equal to `Height * Width * Depth` of the second input

## Outputs

Single output returns the blob with multiplication results, of the dimensions:

- `BatchLength`, `BatchWidth`, `ListSize`, `Height`, `Width`, `Depth` are equal to these dimensions of the first input
- `Channels` is equal to the `Channels` of the second input.
