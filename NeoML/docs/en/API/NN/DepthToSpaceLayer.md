# CDepthToSpaceLayer Class

<!-- TOC -->

- [CDepthToSpaceLayer Class](#cdepthtospacelayer-class)
    - [Settings](#settings)
        - [Block size](#block-size)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that transforms each pixel of the image from input set into square blocks `GetBlockSize() x GetBlockSize()`.

This operation is the inverse function of [CSpaceToDepthLayer](SpaceToDepthLayer.md).

## Settings

### Block size

```c++
void SetBlockSize( int blockSize );
```

Sets the value by which the image size will be divided in the final result. The image channels should be a multiple of the square of this value. The value should be greater than `1`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with the images, of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to the number of images
- `Height` is the image height
- `Width` is the image width
- `Depth` is equal to `1`
- `Channels` is the number of channels in the image format; should be a multiple of `GetBlockSize() * GetBlockSize()`

## Outputs

The single output contains a blob with the resulting images, of the dimensions:

- `BatchLength` is equal to the input `BatchLength`
- `BatchWidth` is equal to the input `BatchWidth`
- `ListSize` is equal to the input `ListSize`
- `Height` is equal to the input `Height * GetBlockSize()`
- `Width` is equal to the input `Width * GetBlockSize()`
- `Depth` is equal to `1`
- `Channels` is equal to the input `Channels / ( GetBlockSize() * GetBlockSize() )`.
