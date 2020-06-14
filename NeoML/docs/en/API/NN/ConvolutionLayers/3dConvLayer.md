# C3dConvLayer Class

<!-- TOC -->

- [C3dConvLayer Class](#c3dconvlayer-class)
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

This class implements a layer that performs convolution on a set of three-dimensional multi-channel images. Zero-padding is supported.

## Settings

### Filters size

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
void SetFilterDepth( int filterDepth );
void SetFilterCount( int filterCount );
```

Sets the filters' size and number.

### Convolution stride

```c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
void SetStrideDepth( int strideDepth );
```

Sets the convolution stride. By default, the stride is `1`.

### Padding

```c++
void SetPaddingHeight( int paddingHeight );
void SetPaddingWidth( int paddingWidth );
void SetPaddingDepth( int paddingDepth );
```

Sets the width, height, and depth of zero-padding that will be added around the image. For example, if you set the padding width to 1, two additional rectangles filled with zeros will be added to the image: one on the left and one on the right.

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

- `BatchLength * BatchWidth * ListSize` is equal to the number of filters used (`GetFilterCount()`)
- `Height` is equal to `GetFilterHeight()`
- `Width` is equal to `GetFilterWidth()`
- `Depth` is equal to `GetFilterDepth()`
- `Channels` is equal to the inputs' `Channels`

### Free terms

```c++
CPtr<CDnnBlob> GetFreeTermData() const;
```

The free terms are represented by a blob of the total size equal to the number of filters used (`GetFilterCount()`).

## Inputs

Each input accepts a blob with several images. The dimensions of all inputs should be the same:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set.
- `Height` - the images' height.
- `Width` - the images' width.
- `Depth` - the images' depth.
- `Channels` - the number of channels the image format uses.

## Outputs

For each input the layer has one output. It contains a blob with the result of the convolution. The output blob dimensions are:

- `BatchLength` is equal to the input `BatchLength`.
- `BatchWidth` is equal to the input `BatchWidth`.
- `ListSize` is equal to the input `ListSize`.
- `Height` can be calculated from the input `Height` as
`(2 * PaddingHeight + Height - (1 + DilationHeight * (FilterHeight - 1)))/StrideHeight + 1`.
- `Width` can be calculated from the input `Width` as
`(2 * PaddingWidth + Width - (1 + DilationWidth * (FilterWidth - 1)))/StrideWidth + 1`.
- `Depth` can be calculated from the input `Depth` as  
`(2 * PaddingDepth + Depth - FilterDepth)/StrideDepth + 1`.
- `Channels` is equal to `GetFilterCount()`.
