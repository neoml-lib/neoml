# CRepeatSequenceLayer Class

<!-- TOC -->

- [CRepeatSequenceLayer Class](#crepeatsequencelayer-class)
    - [Settings](#settings)
        - [Repetitions count](#repetitions-count)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that repeats the input sequence several times.

## Settings

### Repetitions count

```c++
void SetRepeatCount(int count);
```

Sets the number of repetitions. For example: you set it to `2`, a `{0,3,7}` sequence will be transformed into `{0,3,7,0,3,7}`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with a sequence of objects, of the dimensions:

- `BatchLength` is the sequence length
- `BatchWidth * ListSize` is the number of sequences in the set
- `Height * Width * Depth * Channels` is the size of each object in the sequence

## Outputs

The single output returns a blob with the results, of the dimensions:

- `BatchLength` is `GetRepeatCount()` times larger than the input `BatchLength`.
- the other dimensions are equal to the corresponding input dimensions.
