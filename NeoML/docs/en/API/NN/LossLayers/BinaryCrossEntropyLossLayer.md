# CBinaryCrossEntropyLossLayer Class

<!-- TOC -->

- [CBinaryCrossEntropyLossLayer](#cbinarycrossentropylosslayer-class)
    - [Settings](#settings)
        - [Correct classification weight](#correct-classification-weight)
        - [Loss weight](#loss-weight)
        - [Gradient clipping](#gradient-clipping)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Getting the value of the loss function](#getting-the-value-of-the-loss-function)

<!-- /TOC -->

This class implements a layer that calculates a cross-entropy loss function for binary classification.

The function is calculated according to the formula:


```c++
loss = y * -log(sigmoid(x)) + (1 - y) * -log(1 - sigmoid(x))
```

where:

- `x` is the network response.
- `y` is the correct class label (can be `1` or `-1`).

Please note that this function first calculates a `sigmoid` on the network response. It is best not to connect this layer input to the output of another `sigmoid`-calculating layer.

## Settings

### Correct classification weight

```c++
void SetPositiveWeight( float value );
```

Sets the multiplier for the term that corresponds to the objects for which the class has been detected correctly. You can tune this value to prioritize precision (set `value < 1`) or recall (set `value > 1`) during training.

The default value is `1`.


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
