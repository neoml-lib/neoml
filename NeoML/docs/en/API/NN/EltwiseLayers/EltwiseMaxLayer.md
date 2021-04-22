# CEltwiseMaxLayer Class

<!-- TOC -->

- [CEltwiseMaxLayer Class](#celtwisemaxlayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that finds the maximum among the elements that are at the same position in all input blobs.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer should have at least two inputs. The [blobs](../DnnBlob.md) of all inputs should have the same dimensions.

## Outputs

The layer has one output of the same size as each of the inputs. The output blob contains the element-wise maximum of the input blobs.
