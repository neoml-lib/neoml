# CImageToPixelLayer Class

<!-- TOC -->

- [CImageToPixelLayer Class](#cimagetopixellayer-class)
    - [Settings](#settings)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that extracts a set of pixel sequences along the specified coordinates from a set of two-dimensional images.

## Settings

There are no settings for this layer.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer has two inputs. 

The first input accepts a blob with a set of two-dimensional images:

- `BatchLength` is equal to `1`
- `BatchWidth` is the number of sequences in the set
- `ListSize` is equal to `1`
- `Height` is the images' height
- `Width` is the images' width
- `Depth` is equal to `1`
- `Channels` is the number of channels the image format uses

The second input accepts a blob with `int` data that contains lists of pixel coordinates, of the dimensions:

- `BatchWidth` is equal to the first input `BatchWidth`
- `ListSize` is the length of each sequence
- all other dimensions are equal to `1`

The coordinates of a pixel `(col, row)` are represented in this blob by a single integer number equal to `row * GetImageWidth() + col`.

## Outputs

The single output returns a blob with the pixel sequences:

- `BatchLength` is equal to `1`
- `BatchWidth` equals the inputs' `BatchWidth`
- `ListSize` equals the second input `ListSize`
- `Height`, `Width`, and `Depth` are equal to `1`
- `Channels` is equal to the first input `Channels`

The blob contains the pixel sequences taken from the images of the first input using the coordinates from the second input.
