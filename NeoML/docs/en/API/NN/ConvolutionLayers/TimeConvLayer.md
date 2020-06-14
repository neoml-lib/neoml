# CTimeConvLayer Class

<!-- TOC -->

- [CTimeConvLayer](#ctimeconvlayer-class)
    - [Settings](#settings)
        - [Filters size](#filters-size)
        - [Convolution stride](#convolution-stride)
        - [Padding](#padding)
        - [Dilated convolution](#dilated-convolution)
    - [Trainable parameters](#trainable-parameters)
        - [Filters](#filters)
        - [Free terms](#free-terms)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that performs time convolution on a set of sequences. Zero-padding and dilated convolution are supported.

## Settings

### Filters size

```c++
void SetFilterSize( int filterSize );
void SetFilterCount( int filterCount );
```

Set up the filters' size and number.

### Convolution stride

```c++
void SetStride( int stride );
```

Sets the convolution stride. The default value is `1`.

### Padding

```c++
void SetPadding( int padding );
```

Specifies how many zero elements should be added at either end of a sequence before performing convolution. For example, if you set `SetPadding( 1 );` two zeros will be added to the sequence - one at the start and one at the finish. The default value is `0`, that is, no padding used.

### Dilated convolution

```c++
void SetDilation( int dilation );
```

Sets the step value for dilated convolution. Dilated convolution applies the filter not to the consecutive elements of the original sequence but to the elements with the gaps of specified size between them.

By default, this value is equal to `1`: no dilation, consecutive elements are used.

## Trainable parameters

### Filters

```c++
CPtr<CDnnBlob> GetFilterData() const;
```

The filters  are represented by a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength` is equal to `1`
- `BatchWidth` is equal to `GetFilterCount()`
- `Height` is equal to `GetFilterSize()`
- `Width` is equal to `1`
- `Depth` is equal to `1`
- `Channels` is equal to the inputs' `Height * Width * Depth * Channels`

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const;
```

The free terms are represented by a blob of the total size equal to the number of filters used (`GetFilterCount()`).

## Inputs

Each input accepts a blob with several sequences. The dimensions of all inputs should be the same:

- `BatchLength` - the sequences' length
- `BatchWidth * ListSize` - the number of sequences in the set
- `Height * Width * Depth * Channels` - the size of each element in the sequences.

## Outputs 

For each input the layer has one output. It contains a blob with the result of the convolution. The output blob dimensions are:

- `BatchLength` can be calculated from the input as
`(2 * Padding + BatchLength - (1 + Dilation * (FilterSize - 1)))/Stride + 1`.
- `BatchWidth` is equal to the inputs' `BatchWidth`.
- `ListSize` is equal to the inputs' `ListSize`.
- `Height` is equal to `1`.
- `Width` is equal to `1`.
- `Depth` is equal to `1`.
- `Channels` is equal to `GetFilterCount()`.
