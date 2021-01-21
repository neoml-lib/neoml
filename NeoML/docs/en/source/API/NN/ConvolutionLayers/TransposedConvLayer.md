# CTransposedConvLayer Class

<!-- TOC -->

- [CTransposedConvLayer Class](#ctransposedconvlayer-class)
    - [Settings](#settings)
        - [Filters size](#filters-size)
        - [Convolution stride](#convolution-stride)
        - [Padding](#padding)
        - [Dilated convolution](#dilated-convolution)
        - [Using the free terms](#using-the-free-terms)
    - [Trainable parameters](#trainable-parameters)
        - [Filters](#filters)
        - [Free terms](#free-terms)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that performs transposed convolution (sometimes also called *deconvolution* or *up-convolution*) on a set of two-dimensional multi-channel images. Padding and dilated convolution are supported.

## Settings

### Filters size

~~~c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
void SetFilterCount( int filterCount );
~~~

Sets the filters' size and number.

### Convolution stride

~~~c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
~~~

Sets the convolution stride. By default, the stride is `1`.

### Padding

~~~c++
void SetPaddingHeight( int paddingHeight );
void SetPaddingWidth( int paddingWidth );
~~~

Sets the width and height of padding that should be removed from the convolution result. For example, if `SetPaddingWidth( 1 );`, two columns - one on the right and one on the left - will be cut off of the resulting image. By default these values are set to `0`.

### Dilated convolution

~~~c++
void SetDilationHeight( int dilationHeight );
void SetDilationWidth( int dilationWidth );
~~~

Sets the vertical and horizontal step values for dilated convolution. Dilated convolution applies the filter not to the consecutive pixels of the original image but to pixels with the gaps between.

By default, these values are equal to `1`: no dilation, consecutive pixels are used.

### Using the free terms

```c++
void SetZeroFreeTerm(bool isZeroFreeTerm);
```

Specifies if the free terms should be used. If you set this value to `true`, the free terms vector will be set to all zeros and won't be trained. By default, this value is set to `false`.

## Trainable parameters

### Filters

~~~c++
CPtr<CDnnBlob> GetFilterData() const;
~~~

The filters are represented by a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength` is equal to `1`
- `BatchWidth` is equal to the inputs' `Channels * Depth`
- `ListSize` is equal to `1`
- `Height` is equal to `GetFilterHeight()`
- `Width` is equal to `GetFilterWidth()`
- `Depth` is equal to `1`
- `Channels` is equal to `GetFilterCount()`

### Free terms

~~~c++
CPtr<CDnnBlob> GetFreeTermData() const;
~~~

The free terms are represented by a blob of the total size equal to the number of filters used (`GetFilterCount()`).

## Inputs

Each input accepts a blob with several images. The dimensions of all inputs should be the same:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set.
- `Height` - the images' height.
- `Width` - the images' width.
- `Depth * Channels` - the number of channels the image format uses.

## Outputs

For each input the layer has one output. It contains a blob with the result of convolution. The output blob dimensions are:

- `BatchLength` is equal to the input `BatchLength`.
- `BatchWidth` is equal to the input `BatchWidth`.
- `ListSize` is equal to the input `ListSize`.
- `Height` can be calculated from the input `Height` as
`StrideHeight * (Height - 1) + (FilterHeight - 1) * DilationHeight + 1 - 2 * PaddingHeight`.
- `Width` can be calculated from the input `Width` as
`StrideWidth * (Width - 1) + (FilterWidth - 1) * DilationWidth + 1 - 2 * PaddingWidth`.
- `Depth` is equal to `1`.
- `Channels` is equal to `GetFilterCount()`.
