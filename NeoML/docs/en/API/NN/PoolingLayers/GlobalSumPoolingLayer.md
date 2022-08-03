# CGlobalSumPoolingLayer Class

<!-- TOC -->

- [CGlobalSumPoolingLayer Class](#cglobalsumpoolinglayer-class)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer which performs sum pooling on `Height`, `Width`, and `Depth` dimensions of the input.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a [blob](../DnnBlob.md) of any data type of the following dimensions:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set
- `Height` - the images' height
- `Width` - the images' width
- `Depth` - the images' depth
- `Channels` - the number of channels the image format uses

## Outputs

The single output contains a blob of the dimensions:

- `BatchLength` is equal to the input `BatchLength`
- `BatchWidth` is equal to the input `BatchWidth`
- `ListSize` is equal to the input `ListSize`
- `Height` is equal to `1`
- `Width` is equal to `1`
- `Depth` is equal to `1`
- `Channels` is equal to the input `Channels`

