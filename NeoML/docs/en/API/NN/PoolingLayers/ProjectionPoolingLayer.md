# CProjectionPoolingLayer Class

<!-- TOC -->

- [CProjectionPoolingLayer Class](#cprojectionpoolinglayer-class)
    - [Settings](#settings)
        - [Projection dimension](#projection-dimension)
        - [Filter stride](#restore-original-image-size)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#Outputs)

<!-- /TOC -->

This class implements a layer which performs mean pooling over one of the blob dimension.

## Settings

### Projection dimension

```c++
// Projection dimension
void SetDimenion( TBlobDim dimension );
```

The default value is `BD_Width`.

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

The single output contains a blob with the results.

If `GetRestoreOriginalImageSize` is `true` then output is of the same size as input.

If `GetRestoreOriginalImageSize` is `false` then the projection dimension of output size is equal to `1` and the rest of the dimensions are equal to the ones of the input.
