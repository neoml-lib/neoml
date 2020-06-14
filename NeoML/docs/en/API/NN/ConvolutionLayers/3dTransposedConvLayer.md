# C3dTransposedConvLayer Class

<!-- TOC -->

- [C3dTransposedConvLayer Class](#c3dtransposedconvlayer-class)
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

This class implements a layer that performs transposed convolution (sometimes also called *deconvolution* or *up-convolution*) on a set of three-dimensional multi-channel images. Padding is supported.

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
void SetStrideDepht( int strideDepth );
```

Sets the convolution stride. By default, the stride is `1`.

### Padding

```c++
void SetPaddingHeight( int paddingHeight );
void SetPaddingWidth( int paddingWidth );
void SetPaddingDepth( int paddingDepth );
```

Sets the width, height, and depth of padding that should be removed from the convolution result. For example, if `SetPaddingWidth( 1 );`, two rectangular sheets - one on the right and one on the left - will be cut off of the resulting image. By default these values are set to `0`.

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

- `BatchLength` is equal to `1`
- `BatchWidth` is equal to the inputs' `Channels`
- `ListSize` is equal to `1`
- `Height` is equal to `GetFilterHeight()`
- `Width` is equal to `GetFilterWidth()`
- `Depth` is equal to `GetFilterDepth()`
- `Channels` is equal to `GetFilterCount()`

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

For each input the layer has one output. It contains a blob with the result of convolution. The output blob dimensions are:

- `BatchLength` is equal to the input `BatchLength`.
- `BatchWidth` is equal to the input `BatchWidth`.
- `ListSize` is equal to the input `ListSize`.
- `Height` can be calculated from the input `Height` as
`StrideHeight * (Height - 1) + FilterHeight - 2 * PaddingHeight`.
- `Width` can be calculated from the input `Width` as
`StrideWidth * (Width - 1) + FilterWidth - 2 * PaddingWidth`.
- `Depth` can be calculated from the input `Depth` as  
`StrideDepth * (Depth - 1) + FilterDepth - 2 * PaddingDepth`.
- `Channels` is equal to `GetFilterCount()`.
