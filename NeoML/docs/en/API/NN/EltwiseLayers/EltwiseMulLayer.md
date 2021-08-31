# CEltwiseMulLayer Class

<!-- TOC -->

- [CEltwiseMulLayer Class](#celtwisemullayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that multiplies its inputs element by element.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer should have at least two inputs. The [blobs](../DnnBlob.md) of all inputs should have the same dimensions.

## Outputs

The layer has one output of the same size as each of the inputs. The output blob contains the element-wise product of the input blobs.
