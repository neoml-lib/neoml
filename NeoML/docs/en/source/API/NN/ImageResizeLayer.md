# CImageResizeLayer Class

<!-- TOC -->

- [CImageResizeLayer Class](#cimageresizelayer-class)
    - [Settings](#settings)
        - [Resize settings](#resize-settings)
        - [Values to fill in](#values-to-fill-in)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer which resizes a set of two-dimensional multi-channel images.

## Settings

### Resize settings

```c++
// The side of the image that will be added to or deleted from
enum TImageSide {
    IS_Left = 0, // left
    IS_Right, // right
    IS_Top, // top
    IS_Bottom, // bottom

    IS_Count,
};

void SetDelta( TImageSide side, int delta );
```

Specifies the difference between the original and the resized image. If the difference is negative, the corresponding amount of columns or rows will be deleted from the specified side of the image. If the difference is greater than 0, the columns or rows will be added.

The default value is `0`: the image will not be resized.

### Values to fill in

```c++
void SetDefalutValue( float value );
```

Sets the value with which new pixels will be filled. The default is `0`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with a set of images, of the dimensions:

- `BatchLength * BatchWidth * ListSize` - the number of images
- `Height` - the images' height
- `Width` - the images' width
- `Depth * Channels` - the number of channels the image format uses

## Outputs

The single output contains a blob with the resized images. Its dimensions will be:

- `BatchLength`, `BatchWidth`, `ListSize`, `Depth`, and `Channels` are equal to the input dimensions.
- `Height` is equal to `Height + GetDelta( IS_Top ) + GetDelta( IS_Bottom )` calculated from the input `Height`.
- `Width` is equal to `Width + GetDelta( IS_Left ) + GetDelta( IS_Right )` calculated from the input `Width`.
