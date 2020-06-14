# CGlobalMaxPoolingLayer Class

<!-- TOC -->

- [CGlobalMaxPoolingLayer Class](#cglobalmaxpoolinglayer-class)
    - [Settings](#settings)
        - [The number of maximum elements](#the-number-of-maximum-elements)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer which performs max pooling on `Height`, `Width`, and `Depth` dimensions of the input, allowing for multiple largest elements to be found.

If you set the number of largest elements to `1`, this layer will function exactly as the [`C3dMaxPoolingLayer`](3dMaxPoolingLayer.md) with the filter size equal to the input size.

## Settings

### The number of maximum elements

```c++
void SetMaxCount(int enumSize);
```

Sets the number of largest elements that may be found. Note that these do not have to be equal to each other; the top `GetMaxCount();` elements will be returned.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set
- `Height` - the images' height
- `Width` - the images' width
- `Depth` - the images' depth
- `Channels` - the number of channels the image format uses

## Outputs

The layer may have one or two outputs. The first output contains the values found, in a blob of the dimensions:

- `BatchLength` is equal to the input `BatchLength`
- `BatchWidth` is equal to the input `BatchWidth`
- `ListSize` is equal to the input `ListSize`
- `Height` is equal to `1`
- `Width` is equal to `GetMaxCount()`
- `Depth` is equal to `1`
- `Channels` is equal to the input `Channels`

The second output is optional. It has the same dimensions and contains the (integer) indices of the maximum values in the original blob.
