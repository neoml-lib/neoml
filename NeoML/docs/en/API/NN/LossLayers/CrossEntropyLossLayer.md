# CCrossEntropyLossLayer Class

<!-- TOC -->

- [CCrossEntropyLossLayer Class](#ccrossentropylosslayer-class)
    - [Settings](#settings)
        - [Using Softmax](#using-softmax)
        - [Loss weight](#loss-weight)
        - [Gradient clipping](#gradient-clipping)
        - [Using the second input to calculate gradients](#using-the-second-input-to-calculate-gradients)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Getting the value of the loss function](#getting-the-value-of-the-loss-function)

<!-- /TOC -->

This class implements a layer that calculates a cross-entropy loss function for a classification scenario with multiple classes.

The function is calculated according to the formula:

```c++
loss = -sum(y_i * log(z_i))
```

where:

- `i` iterates over the classes.
- `z_i`, depending on the layer settings:
  - if `IsSoftmaxApplied()` it is the `i`th element of the result of `softmax` function calculated over the network response.
  - otherwise it is the `i`th element of the network response.
- `y_i` represents the class label (that is, it is `1` if the element belongs to the `i` class and `0` if it does not).

Please note that you may set a `softmax` function to be applied inside this function. In this case we recommend that you avoid connecting the first input of this layer to the output of another `softmax`-calculating layer.

## Settings

### Using Softmax

```c++
void SetApplySoftmax( bool applySoftmax )
```

Turns on and off `softmax` function calculation on the network response. The default value is `true` - the input will be run through `softmax`.

### Loss weight

```c++
void SetLossWeight( float lossWeight );
```

Sets the multiplier for this function gradient during training. The default value is `1`. You may wish to change the default if you are using several loss functions in your network.

### Gradient clipping

```c++
void SetMaxGradientValue( float maxValue );
```

Sets the upper limit for the absolute value of the function gradient. Whenever the gradient exceeds this limit its absolute value will be reduced to `GetMaxGradientValue()`.

### Using the second input to calculate gradients

```c++
void SetTrainLabels( bool toSet );
```

Turns on and off gradient calculation for the second input, which contains the class labels.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer may have 2 to 3 inputs:

1. The network output for which you are calculating the loss function. It should contain the probability distribution for `BatchLength * BatchWidth * ListSize` objects over `Height * Width * Depth * Channels` classes. If `IsSoftmaxApplied()` is `false`, each element should be greater or equal to `0`, and for each object the sum over `Height * Width * Depth * Channels` dimension should be equal to `1`.
2. The class labels represented by a blob in one of the two formats:
	* the blob contains `float` data, the dimensions are equal to the first input dimensions. It should be filled with zeros, and only the coordinate of the class to which the corresponding object from the  first input belongs should be `1`.
	* the blob contains `int` data with `BatchLength`, `BatchWidth`, and `ListSize` equal to these dimensions of the first input, and the other dimensions equal to `1`. Each object in the blob contains the number of the class to which the corresponding object from the first input belongs.

3. *[Optional]* The objects' weights. This input should have the same `BatchLength`, `BatchWidth`, and `ListSize` dimensions as the first input. `Height`, `Width`, `Depth`, and `Channels` should be equal to `1`.

## Outputs

This layer has no output.

### Getting the value of the loss function

```c++
float GetLastLoss() const;
```

Use this method to get the value of the loss function calculated on the network's last run.
