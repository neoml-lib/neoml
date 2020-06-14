# CBinaryFocalLossLayer Class

<!-- TOC -->

- [Class CBinaryFocalLossLayer](#cbinaryfocallosslayer-class)
    - [Settings](#settings)
        - [Focal force](#focal-force)
        - [Loss weight](#loss-weight)
        - [Gradient clipping](#gradient-clipping)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Getting the value of the loss function](#getting-the-value-of-the-loss-function)

<!-- /TOC -->

This class implements a layer that calculates a focal loss function for binary classification.

The focal loss function is a modified version of cross-entropy loss function in which the objects that are easily distinguished receive smaller penalties. This helps focus on learning the difference between similar-looking elements of different classes.

The function is calculated according to the formula:

```c++
loss = -pow(sigmoid(-y*x), focalForce) * log(1 + e^(-y*x))
```

where:

- `x` is the network response.
- `y` is the correct class label (can be `1` or `-1`).

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

1. The network output for which you are calculating the loss function. `Height`, `Width`, `Depth`, and `Channels` dimensions of this blob should be equal to `1`.
2. A blob of the same size as the first input, containing the class labels (may be `-1` or `1`).
3. *[Optional]* The objects' weights. This blob should have the same dimensions as the first input.

## Outputs

This layer has no output.

### Getting the value of the loss function

```c++
float GetLastLoss() const;
```

Use this method to get the value of the loss function calculated on the network's last run.
