# CEnumBinarizationLayer Class

<!-- TOC -->

- [CEnumBinarizationLayer Class](#cenumbinarizationlayer-class)
    - [Settings](#settings)
        - [Enumeration size](#enumeration-size)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class converts the enumeration values into *one-hot encoding*.

## Settings

### Enumeration size

```c++
void SetEnumSize(int enumSize);
```

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with `int` or `float` data that contains the enumeration values, of the dimensions:

- `Channels` is equal to `1`
- the other dimensions may be of any size

## Outputs

The single output contains a blob with the vectors that one-hot encode the enumeration values. The dimensions of the blob are:

- `Channels` is equal to `GetEnumSize()`
- the other dimensions are the same as for the input
