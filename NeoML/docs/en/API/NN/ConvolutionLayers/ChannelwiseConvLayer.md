# CChannelwiseConvLayer Class

<!-- TOC -->

- [CChannelwiseConvLayer Class](#cchannelwiseconvlayer-class)
    - [Settings](#settings)
        - [Filters size](#filters-size)
        - [Convolution stride](#convolution-stride)
        - [Padding](#padding)
        - [Using the free terms](#using-the-free-terms)
    - [Trainable parameters](#trainable-parameters)
        - [Filters](#filters)
        - [Free terms](#free-terms)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that performs channel-wise convolution on a set of two-dimensional multi-channel images. Each channel of the input blob is convolved with the corresponding channel of the single filter. Padding is supported.

## Settings

### Filters size

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
void SetFilterCount( int filterCount );
```

Sets the filters' size and number.

### Convolution stride

```c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
```

Sets the convolution stride. By default, the stride is `1`.

### Padding

```c++
void SetPaddingHeight( int paddingHeight );
void SetPaddingWidth( int paddingWidth );
```

Sets the width and height of zero-padding that will be added around the image. For example, if you set the padding width to 1, two additional columns filled with zeros will be added to the image: one on the left and one on the right.

By default, no padding is used, and these values are equal to `0`.

### Using the free terms

```c++
void SetZeroFreeTerm(bool isZeroFreeTerm);
```

Specifies if the free terms should be used. If you set this value to `true`, the free terms vector will be set to all zeros and won't be trained. By default, this value is set to `false`.

## Trainable parameters

### Filters

```c++
CPtr<CDnnBlob> GetFilterData() const;
```

The filters are represented by a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength`, `BatchWidth`, `ListSize` are equal to `1`.
- `Height` is equal to `GetFilterHeight()`.
- `Width` is equal to `GetFilterWidth()`.
- `Depth` is equal to `1`.
- `Channels` is equal to the inputs' `Channels`.

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const;
```

The free terms are represented by a blob of the total size equal to the inputs' channels.

## Inputs

Each input accepts a blob with several images. The dimensions of all inputs should be the same:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set.
- `Height` - the images' height.
- `Width` - the images' width.
- `Channels` - the number of channels the image format uses.

## Outputs

For each input the layer has one output. It contains a blob with the result of the convolution. The output blob dimensions are:

- `BatchLength` is equal to the input `BatchLength`.
- `BatchWidth` is equal to the input `BatchWidth`.
- `ListSize` is equal to the input `ListSize`.
- `Height` can be calculated from the input `Height` as
`(2 * PaddingHeight + Height - FilterHeight)/StrideHeight + 1`.
- `Width` can be calculated from the input `Width` as
`(2 * PaddingWidth + Width - FilterWidth)/StrideWidth + 1`.
- `Depth` is equal to `1`.
- `Channels` is equal to `GetFilterCount()`.
