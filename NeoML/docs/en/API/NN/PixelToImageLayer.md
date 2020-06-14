# CPixelToImageLayer Class

<!-- TOC -->

- [CPixelToImageLayer Class](#cpixeltoimagelayer-class)
    - [Settings](#settings)
        - [Resulting image height](#resulting-image-height)
        - [Resulting image width](#resulting-image-width)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that creates a set of two-dimensional images using a set of pixel sequences with specified coordinates.

## Settings

### Resulting image height

```c++
void SetImageHeight( int newHeight );
```

Sets the height of the output images.

### Resulting image width

```c++
void SetImageWidth( int newWidth );
```

Sets the width of the output images.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer has two inputs.

The first accepts a blob with pixel sequences, of the dimensions:

- `BatchLength` is equal to `1`
- `BatchWidth` is the number of sequences in the set
- `ListSize` is the length of each sequence
- `Height`, `Width`, and `Depth` are equal to `1`
- `Channels` is the number of channels the pixels sequences (and the output images) use.

The second input accepts a blob with `int` data that contains lists of pixel coordinates, of the dimensions:

- `BatchWidth` equals first input `BatchWidth`
- `ListSize` equals first input `ListSize`
- the other dimensions equal `1`

The coordinates of a pixel `(col, row)` are represented in this blob by a single integer number equal to `row * GetImageWidth() + col`.

## Outputs

The single output returns a blob of the dimensions:

- `BatchLength` equals `1`
- `BatchWidth` equals inputs' `BatchWidth`
- `ListSize` equals `1`
- `Height` equals `GetImageHeight()`
- `Width` equals `GetImageWidth()`
- `Depth` equals `1`
- `Channels` equals the first input `Channels`

The blob contains the set of images with the pixel sequences written into the specified coordinates. The pixels that were not given are filled with zeros.
