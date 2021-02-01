# CMaxOverTimePoolingLayer Class

<!-- TOC -->

- [CMaxOverTimePoolingLayer Class](#cmaxovertimepoolinglayer-class)
    - [Settings](#settings)
        - [Filter size](#filter-size)
        - [Filter stride](#filter-stride)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer which performs max pooling on the set of sequences, with the filter applied on `BatchLength` axis.

## Settings

### Filter size

```c++
void SetFilterLength(int length);
```

Sets the filter length. If you set the length to `0` or less, the maximum over the whole axis will be calculated.

### Filter stride

```c++
void SetStrideLength(int length);
```

Sets the filter stride. This value is meaningful only if the filter length is greater than `0`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength` - the sequence length
- `BatchWidth * ListSize` - the number of sequences in the set
- `Height * Width * Depth * Channels` - the length of each vector in the sequences

## Outputs

The single output contains a blob with the results. 

If the filter length is positive, the `BatchLength` of the resulting  blob may be calculated as 
`(BatchLength - GetFilterLength()) / GetStrideLength() + 1`.

If the filter length is equal or less than `0`, the `BatchLength` of the resulting blob is `1`.

The rest of the dimensions of the output blob are equal to the corresponding dimensions of the input.
