# CCtcDecodingLayer Class

<!-- TOC -->

- [CCtcDecodingLayer Class](#cctcdecodinglayer-class)
    - [Settings](#settings)
        - [Blank label for spaces](#blank-label-for-spaces)
        - [Spaces cutoff](#spaces-cutoff)
        - [Arcs cutoff](#arcs-cutoff)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [The most probable sequence](#the-most-probable-sequence)
        - [Linear division graph](#linear-division-graph)

<!-- /TOC -->

The class implements a layer that is looking for the most probable sequences in the response of a [connectionist temporal classification network](README.md).

## Settings

### Blank label for spaces

```c++
void SetBlankLabel( int blankLabel );
```

Sets the value of the "blank" label that will be used as the space between other labels.

### Spaces cutoff

```c++
void SetBlankProbabilityThreshold( float threshold );
```

Sets the probability threshold for blank labels, when building a linear division graph (LDG).

### Arcs cutoff

```c++
void SetArcProbabilityThreshold( float threshold );
```

Sets the probability threshold for cutting off arcs when building an LDG.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer has one or two inputs:

1. The first input accepts a blob with the network response, of the dimensions:
    * `BatchLength` is the maximum response sequence length
    * `BatchWidth` is the number of sequences in the set
    * `ListSize` is equal to `1`
    * `Height * Width * Depth * Channels` is the number of classes
2. *[Optional]* The second input accepts a blob with `int` data that contains the network response sequences' lengths. If this input is not connected, the network response sequences are considered to be the first input's `BatchLength` long. This input dimensions are:
    * `BatchWidth` equal to the first input `BatchWidth`
    * the other dimensions are equal to `1`

## Outputs

The layer has no outputs.

### The most probable sequence

```c++
void GetBestSequence(int sequenceNumber, CArray<int>& bestLabelSequence) const;
```

Retrieves the most probable sequence for the object with the  `sequenceNumber` index in the set.

### Linear division graph

```c++
bool BuildGLD(int sequenceNumber, CLdGraph<CCtcGLDArc>& gld) const;
```

Retrieves the linear division graph for the object with the `sequenceNumber` index in the set.
