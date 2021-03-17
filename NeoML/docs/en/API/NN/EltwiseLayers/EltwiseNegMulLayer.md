# CEltwiseNegMulLayer Class

<!-- TOC -->

- [CEltwiseNegMulLayer Class](#celtwisenegmullayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that calculates the element-wise product of `1 - x`, where `x` is the element of the first input, and the corresponding elements of all other inputs.

## Settings

This layer has no settings.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer should have at least two inputs. The [blobs](../DnnBlob.md) of all inputs should have the same dimensions.

## Outputs

The layer has one output of the same size as each of the inputs.
