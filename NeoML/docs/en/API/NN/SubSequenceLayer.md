# CSubSequenceLayer Class

<!-- TOC -->

- [CSubSequenceLayer Class](#csubsequencelayer-class)
    - [Settings](#settings)
        - [Subsequence start](#subsequence-start)
        - [Subsequence length](#subsequence-length)
        - [Reverse the original sequence](#reverse-the-original-sequence)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that extracts a subsequence from each vector sequence of the set.

## Settings

### Subsequence start

```c++
void SetStartPos(int startPos);
```

Sets the position of the first element of the subsequence. For `startPos >= 0`, the position will be counted from the start of the original sequence. For `startPos < 0` it will be counted from the end, with `-1` standing for the last element of the original sequence.

### Subsequence length

```c++
void SetLength(int length);
```

Sets the length of the subsequence to be extracted. If `length > 0`, the order of elements is the same as in the original. If `length < 0`, the order is reversed.

The length of the subsequence actually extracted will be not greater than `abs(length)`, but may be smaller if the specified length would not fit into the original after starting at the specified position.

### Reverse the original sequence

```c++
void SetReverse();
```

Sets the length and starting position so that the original will be reversed as a whole, without changing its length.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with a set of objects; the objects are numbered along the `BatchLength` dimension of this blob.

## Outputs

The single output returns a blob with the subsequence of objects, of the following dimensions:

- `BatchLength` is equal to the smaller value out of `abs(GetLength()` and the maximum length of the subsequence for which it still fits into the original.
- The rest of the dimensions are equal to the input dimensions.
