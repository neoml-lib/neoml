# CProjectionPoolingLayer Class

<!-- TOC -->

- [CProjectionPoolingLayer Class](#cprojectionpoolinglayer-class)
    - [Settings](#settings)
        - [Projection direction](#projection-direction)
        - [Filter stride](#restore-original-image-size)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#Outputs)

<!-- /TOC -->

This class implements a layer which performs mean pooling over blob height or width.

## Settings

### Projection direction

```c++
// Projection direction
enum TDirection {
    // Along BD_Width
    D_ByRows,
    // Along BD_Height
    D_ByColumns,

    D_EnumSize
};

// Projection direction
void SetDirection( TDirection _direction );
```

The default value is `D_ByRows`.

### Restore original image size

```c++
void SetRestoreOriginalImageSize( bool flag );
```

If `true` then output blob will be of the same size as input blob, and mean values will be broadcasted along pooling direction.
If `false` then output blob size along pooling direction will be `1`.
The default value is `false`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a [blob](../DnnBlob.md) of the following dimensions:

- `BatchLength * BatchWidth * ListSize` - the number of images in the set
- `Height` - images' height
- `Width` - images' width
- `Depth * Channels` - the number of channels the image format uses

## Outputs

The single output contains a blob of the dimensions:

- `BatchLength` is equal to the input `BatchLength`.
- `BatchWidth` is equal to the input `BatchWidth`.
- `ListSize` is equal to the input `ListSize`.
- `Height` is equal to `1` if `RestoreOriginalImageSize` is `false` and the projection is `D_ByColumns`. Equals to `Height` otherwise.
- `Width` is equal to `1` if `RestoreOriginalImageSize` is `false` and the projection is `D_ByRows`. Equals to `Width` otherwise.
- `Depth` is equal to the input `Depth`.
- `Channels` is equal to the input `Channels`.
