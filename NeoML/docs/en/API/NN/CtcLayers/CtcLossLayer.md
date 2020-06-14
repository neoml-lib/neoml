# CCtcLossLayer Class

<!-- TOC -->

- [CCtcLossLayer Class](#cctclosslayer-class)
    - [Settings](#settings)
        - [Blank label for spaces](#blank-label-for-spaces)
        - [Skipping blanks](#skipping-blanks)
        - [Loss weight](#loss-weight)
        - [Gradient clipping](#gradient-clipping)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Getting the value of the loss function](#getting-the-value-of-the-loss-function)

<!-- /TOC -->

This class implements a layer that calculates the loss function used for connectionist temporal classification ([CTC](README.md)).

## Settings

### Blank label for spaces

```c++
void SetBlankLabel( int blankLabel );
```

Sets the value of the "blank" label that will be used as the space between other labels.

### Skipping blanks

```c++
void SetAllowBlankLabelSkips( bool enabled );
```

Sets the flag that allows skipping the blank labels when aligning.

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

The layer may have two to five inputs:

1. The first input accepts a blob with the network response, of the dimensions:
    * `BatchLength` is the maximum sequence length in the response
    * `BatchWidth` is the number of sequences in the set
    * `ListSize` is equal to `1`
    * `Height * Width * Depth * Channels` is equal to the number of classes
2. The second input accepts a blob with `int` data containing the correct labels, of the dimensions:
    * `BatchLength` is the maximum labels sequence length
    * `BatchWidth` is the number of sequences (should be equal to `BatchWidth` of the first input)
    * the other dimensions are equal to `1`
3. *[Optional]* The third input accepts a blob with `int` data containing the label sequences' lengths. If this input is not connected, the label sequences are considered to be the second input's `BatchLength` long. This input dimensions are:
    * `BatchWidth` equal to the first input `BatchWidth`
    * the other dimensions are equal to `1`
4. *[Optional]* The fourth input accepts a blob with `int` data that contains the network response sequences' lengths. If this input is not connected, the network response sequences are considered to be the first input's `BatchLength` long. This input dimensions are:
    * `BatchWidth` equal to the first input `BatchWidth`
    * the other dimensions are equal to `1`
5. *[Optional]* The fifth input accepts a blob with the sequences' weights, of the dimensions:
    * `BatchWidth` equal to the first input `BatchWidth`
    * the other dimensions are equal to `1`

## Outputs

This layer has no output.

### Getting the value of the loss function

```c++
float GetLastLoss() const;
```

Use this method to get the value of the loss function calculated on the network's last run.
