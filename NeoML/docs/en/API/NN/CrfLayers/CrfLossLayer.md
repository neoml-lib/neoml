# CCrfLossLayer Class

<!-- TOC -->

- [CCrfLossLayer Class](#ccrflosslayer-class)
    - [Settings](#settings)
        - [Loss weight](#loss-weight)
        - [Gradient clipping](#gradient-clipping)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Getting the value of the loss function](#getting-the-value-of-the-loss-function)

<!-- /TOC -->

This class implements a layer that calculates the loss function used for training a CRF. The value of the loss function is equal to `-log(probability of the correct class sequence)`

## Settings

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

The layer has no trainable parameters.

## Inputs

The layer has 3 mandatory inputs and an optional one.

1. The first input accepts a blob with `int` data that contains optimal class sequences. The dimensions of the blob are:

    - `BatchLength` is equal to the network inputs' `BatchLength`
    - `BatchWidth` is equal to the network inputs' `BatchWidth`
    - `Channels` is the number of classes
    - the other dimensions are equal to `1`

2. The second input accepts a blob with `float` data containing non-normalized logarithm of probabilities of the optimal class sequences. The blob has the same dimensions as the first input.

3. The third input accepts a blob with non-normalized logarithm of probability of the correct class being in this position. The blob dimensions are:

    - `BatchLength` is equal to the network inputs' `BatchLength`
    - `BatchWidth` is equal to the network inputs' `BatchWidth`
    - the other dimensions are equal to `1`

4. *[Optional]* The fourth input accepts a blob with the sequences' weights, of dimensions:
    - `BatchWidth` is equal to the `BatchWidth` of the first input
    - the other dimensions are equal to `1`

## Outputs

This layer has no output.

### Getting the value of the loss function

```c++
float GetLastLoss() const;
```

Use this method to get the value of the loss function calculated on the network's last run.
