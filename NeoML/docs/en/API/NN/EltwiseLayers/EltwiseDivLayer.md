# CEltwiseDivLayer Class

<!-- TOC -->

- [CEltwiseDivLayer Class](#celtwisedivlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that divides its first input by the second element by element.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer should two inputs. The [blobs](../DnnBlob.md) of those inputs should have the same dimensions.

## Outputs

The layer has one output of the same size as each of the inputs. The output blob contains the element-wise division of the input blobs.
