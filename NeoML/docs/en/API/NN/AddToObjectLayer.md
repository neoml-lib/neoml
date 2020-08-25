# CAddToObjectLayer Class

<!-- TOC -->

- [CAddToObjectLayer Class](#caddtoobjectlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that adds up its inputs element by element.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

This layer has 2 inputs.

The first input accepts a blob containing `BatchLength * BatchWidth * ListSize` objects of size `Height * Width * Depth * Channels`.

The second input accepts a blob of the dimensions:

- `Height`, `Width`, `Depth` and `Channels` are equal to these dimensions of the first input
- all other dimensions are equal to `1`

## Outputs

The single output returns a blob of size equal to the first input. This blob contains the sums of the corresponding objects of the first input with the contents of the second input.
