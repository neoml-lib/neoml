# CEuclideanLossLayer Class

<!-- TOC -->

- [CEuclideanLossLayer Class](#ceuclideanlosslayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Getting the value of the loss function](#getting-the-value-of-the-loss-function)

<!-- /TOC -->

This class implements a layer that calculates a loss function equal to the Euclidean distance between the classes from the network response and the objects belonging to the correct classes.

## Settings

The layer has no settings.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer may have 2 to 3 inputs:

1. The network output for which you are calculating the loss function. It contains `BatchLength * BatchWidth * ListSize` objects, each of `Height * Width * Depth * Channels` size.
2. A blob of the same size as the first input, containing the correct class objects. The loss function will calculate the Euclidean distance between the first and the second input.
3. *[Optional]* The objects' weights. This blob should have the same dimensions as the first input.

## Outputs

This layer has no output.

### Getting the value of the loss function

```c++
float GetLastLoss() const;
```

Use this method to get the value of the loss function calculated on the network's last run.
