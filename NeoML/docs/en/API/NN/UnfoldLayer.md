# CUnfoldLayer Class

<!-- TOC -->

- [CUnfoldLayer Class](#cunfoldlayer-class)
    - [Settings](#settings)
        - [Filters size](#filters-size)
        - [Convolution stride](#convolution-stride)
        - [Padding](#padding)
        - [Dilated convolution](#dilated-convolution)
        - [Using the free terms](#using-the-free-terms)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

Unfold layer extracts data from the regions, that would be affected by the convolution with given parameters.


## Settings

### Filters size

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
```

Sets the filter size.

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

### Dilated convolution

```c++
void SetDilationHeight( int dilationHeight );
void SetDilationWidth( int dilationWidth );
```

Sets the vertical and horizontal step values for dilated convolution. Dilated convolution applies the filter not to the consecutive pixels of the original image but to pixels with the gaps between.

By default, these values are equal to `1`: no dilation, consecutive pixels are used.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

The single input accepts a blob with several images of the following dimensions:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set.
- `Height` - the images' height.
- `Width` - the images' width.
- `Depth * Channels` - the number of channels the image format uses.

## Выходы

The single output contains a blob with the result of the following dimensions:

- `BatchLength` is equal to the input `BatchLength`.
- `BatchWidth` is equal to the input `BatchWidth`.
- `ListSize` is equal to the input `ListSize`.
- `Height` is equal to the product of the convolution's output height and width, where height is equal to `(2 * PaddingHeight + InputHeight - (1 + DilationHeight * (FilterHeight - 1)))/StrideHeight + 1`, and width is equal to `(2 * PaddingWidth + InputWidth - (1 + DilationWidth * (FilterWidth - 1)))/StrideWidth + 1`;
- `Width` is equal to `1`;
- `Depth` is equal to `1`;
- `Channels` is equal to `Channels` of the input multiplied by `filterHeight * filterWidth`.
