# CUpsampling2DLayer Class

<!-- TOC -->

- [CUpsampling2DLayer Class](#cupsampling2dlayer-class)
    - [Settings](#settings)
        - [Height multiplier](#height-multiplier)
        - [Width-multiplier](#width-multiplier)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that scales up a set of two-dimensional multi-channel images. The new pixels are filled up by repeating the existing pixels' values.

## Settings

### Height multiplier

```c++
void SetHeightCopyCount( int newHeightCopyCount );
```

### Width multiplier

```c++
void SetWidthCopyCount( int newHeightCopyCount );
```

## Trainable parameters

The layer has no trainable parameters.

## Inputs

Each of the inputs accepts a blob with a set of images:

- `BatchLength * BatchWidth * ListSize` is the number of images in the set
- `Height` is the images' height
- `Width` is the images' width
- `Depth * Channels` is the number of channels the image format uses

## Outputs

For each input, the corresponding output returns a blob with the upscaled image. Upsampling is performed by repetition of original pixels without any interpolation.

The blob dimensions are:

- `BatchLength`, `BatchWidth`, `ListSize`, `Depth`, and `Channels` equal the input dimensions
- `Height` is `GetHeightCopyCount()` times larger than the input `Height`
- `Width` is `GetWidthCopyCount()` times larger than the input `Width`
