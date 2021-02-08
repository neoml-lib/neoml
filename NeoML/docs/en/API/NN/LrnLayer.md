# CLrnLayer Class

<!-- TOC -->

- [CLrnLayer Class](#clrn-class)
    - [Settings](#settings)
        - [Window size](#window-size)
        - [Bias](#bias)
        - [Scale (alpha)](#scale-(alpha))
        - [Exponent (beta)](#exponent-(beta))
    - [Trainable parameters](#trainable-parameters)
        - [Final values](#final-values)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that performs local response normalization normalization using the following formula:

```c++
LRN(x)[obj][ch] = x[obj][ch] * / ((bias + alpha * sqrSum[obj][ch] / windowSize) ^ beta)
```

where:

- `obj` is index of object `[0; BlobSize / Channels)`
- `ch` is index of channel `[0; Channels)` 
- `windowSize`, `bias`, `alpha`, `beta` are settings
- `sqrSum` is calculated using the following formula:

```c++
sqrSum(x)[obj][ch] = sum(x[obj][i] * x[obj][i] for each i in [ch_min, ch_max])
ch_min = max(0, ch - floor((windowSize - 1)/2))
ch_max = min(C - 1, ch + ceil((windowSize - 1)/2))
```

## Settings

### Window size

```c++
void SetWindowSize( int value );
```

Sets size of the window used during the calculation of `sqrSum`.

### Bias

```c++
void SetBias( float value );
```

Sets the bias value, which is added to the scaled sum of squares.

### Scale (alpha)

```c++
void SetAlpha( float value );
```

Sets the scale value. The sum of squares is multiplied by this value.

### Exponent (beta)

```c++
void SetBeta( float value );
```

Sets the exponent, used in the formula.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

The single input accepts a blob of any size.

## Outputs

The single output contains a blob of the same size with the results of batch normalization.
