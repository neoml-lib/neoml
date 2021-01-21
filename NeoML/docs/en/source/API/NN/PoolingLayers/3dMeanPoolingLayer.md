# C3dMeanPoolingLayer Class

<!-- TOC -->

- [C3dMeanPoolingLayer Class](#c3dmeanpoolinglayer-class)
    - [Settings](#settings)
        - [Filter size](#filter-size)
        - [Filter stride](#filter-stride)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer which performs mean pooling on a set of three-dimensional multi-channel images.

## Settings

### Filter size

```c++
void SetFilterHeight( int filterHeight );
void SetFilterWidth( int filterWidth );
void SetFilterDepth( int filterDepth );
```

Sets the filter size.

### Filter stride

```c++
void SetStrideHeight( int strideHeight );
void SetStrideWidth( int strideWidth );
void SetStrideDepth( int strideDepth );
```

Sets the filter stride. The default value is `1`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set
- `Height` - images' height
- `Width` - images' width
- `Depth` - images' depth
- `Channels` - the number of channels the image format uses

## Outputs

The single output contains a blob of the dimensions:

- `BatchLength` is equal to the input `BatchLength`.
- `BatchWidth` is equal to the input `BatchWidth`.
- `ListSize` is equal to the input `ListSize`.
- `Height` can be calculated from the input `Height` as
`(Height - FilterHeight)/StrideHeight + 1`.
- `Width` can be calculated from the input `Width` as
`(Width - FilterWidth)/StrideWidth + 1`.
- `Depth` can be calculated from the input `Depth` as 
`(Depth - FilterDepth)/StrideDepth + 1`.
- `Channels` is equal to the input `Channels`.
