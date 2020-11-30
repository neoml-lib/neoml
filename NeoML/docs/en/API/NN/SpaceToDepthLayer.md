# CSpaceToDepthLayer Class

<!-- TOC -->

- [CSpaceToDepthLayer Class](#cspacetodepthlayer-class)
    - [Settings](#settings)
        - [Block size](#block-size)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that splits images from a set of two-dimensional multi-channel images into square `GetBlockSize() x GetBlockSize()` blocks and flattens those blocks.

This operation is the inverse function of [CDepthToSpaceLayer](DepthToSpaceLayer.md).

## Settings

### Block size

```c++
void SetBlockSize( int blockSize );
```

Sets the value by which the image size will be multiplied in the final result. The image size along either dimension should be a multiple of this value. The value should be greater than `1`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with the images, of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to the number of images
- `Height` is the image height; should be a multiple of `GetBlockSize()`
- `Width` is the image width; should be a multiple of `GetBlockSize()`
- `Depth` is equal to `1`
- `Channels` is the number of channels in the image format

## Outputs

The single output contains a blob with the resulting images, of the dimensions:

- `BatchLength` is equal to the input `BatchLength`
- `BatchWidth` is equal to the input `BatchWidth`
- `ListSize` is equal to the input `ListSize`
- `Height` is equal to the input `Height / GetBlockSize()`
- `Width` is equal to the input `Width / GetBlockSize()`
- `Depth` is equal to `1`
- `Channels` is equal to the input `Channels * GetBlockSize() * GetBlockSize()`.
