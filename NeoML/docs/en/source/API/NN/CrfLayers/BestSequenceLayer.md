# CBestSequenceLayer Class

<!-- TOC -->

- [CBestSequenceLayer Class](#cbestsequencelayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that finds the optimal class sequence using the output of the [CCrfLayer](CrfLayer.md).
To use it during training the [O_BestPrevClass output computation](CrfLayer.md#O_BestPrevClass-output-computation) must be enabled in the corresponding [CCrfLayer](CrfLayer.md).

## Settings

The layer has no settings.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer has two inputs:

The first input accepts a blob with `int` data that contains the optimal class sequences (i.e. first output of the [CCrfLayer](CrfLayer.md)). The dimensions are:

- `BatchLength` is the sequence length
- `BatchWidth` is the number of sequences in the set
- `Channels` is equal to the number of classes
- the rest of the dimensions are equal to `1`

The second input accepts a blob with `float` data that contains non-normalized logarithm of optimal class sequences probabilities (i.e. the second output of the [CCrfLayer](CrfLayer.md)). It has the same dimensions as the first input.

## Outputs

The single output is a blob with `int` data that contains the optimal class sequences. The blob dimensions are:

- `BatchLength` is equal to the inputs' `BatchLength`
- `BatchWidth` is equal to the inputs' `BatchWidth`
- the other dimensions are equal to `1`
