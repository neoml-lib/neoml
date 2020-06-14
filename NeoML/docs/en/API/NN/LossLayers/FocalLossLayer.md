# CFocalLossLayer Class

<!-- TOC -->

- [CFocalLossLayer Class](#cfocallosslayer-class)
    - [Settings](#settings)
        - [Focal force](#focal-force)
        - [Loss weight](#loss-weight)
        - [Gradient clipping](#gradient-clipping)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Getting the value of the loss function](#getting-the-value-of-the-loss-function)

<!-- /TOC -->

This class implements a layer that calculates a focal loss function for a classification scenario with multiple classes.

The focal loss function is a modified version of cross-entropy loss function in which the objects that are easily distinguished receive smaller penalties. This helps focus on learning the difference between similar-looking elements of different classes.

The function is calculated according to the formula:

```c++
loss = -pow(1 - x_right, focalForce) * log(x_right)
```

where:

- `x_right` is the network response element that represents the probability for the object to belong to the given class.

## Settings

### Focal force

```c++
void SetFocalForce( float value );
```

Sets the focal force, that is, the degree to which learning will concentrate on similar objects. The greater the number, the more focused the learning will become.

This value should be greater than `0`. The default is `2`.

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

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer may have 2 to 3 inputs:

1. The network output for which you are calculating the loss function. It should contain the probability distribution for `BatchLength * BatchWidth * ListSize` objects over `Height * Width * Depth * Channels` classes. Each element should be greater or equal to `0`. For each object, the sum of all elements over `Height * Width * Depth * Channels` dimension should be equal to `1`.
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
