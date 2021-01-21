# CCrfLayer Class

<!-- TOC -->

- [CCrfLayer Class](#ccrflayer class)
    - [Settings](#settings)
        - [The number of classes in the CRF](#the-number-of-classes-in-the-crf)
        - [Empty class](#empty-class)
        - [Variational dropout](#variational-dropout)
        - [O_BestPrevClass output computation](#O_BestPrevClass-output-computation)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that trains and calculates transition probabilities in a conditional random field (CRF).

## Settings

### The number of classes in the CRF

```c++
void SetNumberOfClasses( int numberOfClasses );
```

### Empty class

```c++
void SetPaddingClass( int paddingClass );
```

Sets the number of the empty class used to fill the sequence end.

### Variational dropout

```c++
void SetDropoutRate( float newDropoutRate );
```

### O_BestPrevClass output computation

```c++
void SetBestPrevClassEnabled( bool enabled );
```

Enables computation of the first (`O_BestPrevClass`) output during training.
This setting allows to connect the [CBestSequenceLayer](BestSequenceLayer.md) and get predictions for every batch during training.
Disabled by default.
During inference the output is always computed and this setting is ignored.

## Trainable parameters

The layer trains the transition probabilities, but they are not available to the user.

## Inputs

The layer may have one or two inputs.

The first input accepts a blob with the object sequences, of the dimensions:

- `BatchLength` is the sequence length
- `BatchWidth` is the number of sequences in the set
- `ListSize` is equal to `1`
- `Height * Width * Depth * Channels` is the object size

*[Optional]* The second input accepts a blob with `int` data that contains the correct class sequences. This input is required for training. The blob dimensions are:

- `BatchLength` is equal to the first input `BatchLength`
- `BatchWidth` is equal to the first input `BatchWidth`
- the other dimensions are equal to `1`

## Outputs

The layer may have two to three outputs.

The first output is a blob with `int` data that contains optimal class sequences. The blob dimensions are:

- `BatchLength` is equal to the inputs' `BatchLength`
- `BatchWidth` is equal to the inputs' `BatchWidth`
- `Channels` is equal to `GetNumberOfClasses()`
- the other dimensions are equal to `1`

The second output is a blob with `float` data that contains non-normalized logarithm of optimal class sequences probabilities. It has the same dimensions as the first output.

*[Optional]* The third output will be there only if the layer has two inputs. It contains a blob with non-normalized logarithm of the correct class sequences probabilities. The blob dimensions are:

- `BatchLength` is equal to the inputs' `BatchLength`
- `BatchWidth` is equal to the inputs' `BatchWidth`
- the other dimensions are equal to `1`

The [CBestSequenceLayer](BestSequenceLayer.md) class extracts the best sequence using the first two outputs.
