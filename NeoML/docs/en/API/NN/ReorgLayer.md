# CReorgLayer Class

<!-- TOC -->

- [CReorgLayer Class](#creorglayer-class)
    - [Settings](#settings)
        - [Resize factor](#resize-factor)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that transforms a set of two-dimensional multi-channel images into a set of images of smaller size but with more channels. This operation is used in [YOLO](https://pjreddie.com/darknet/yolo/) architecture.

## Settings

### Resize factor

```c++
void SetStride( int stride );
```

Sets the value by which the image size will be divided in the final result. The image size along either dimension should be a multiple of this value. The value should be greater than `1`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The single input accepts a blob with the images, of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to the number of images
- `Height` is the image height; should be a multiple of `GetStride()`
- `Width` is the image width; should be a multiple of `GetStride()`
- `Depth` is equal to `1`
- `Channels` is the number of channels in the image format

## Outputs

The single output contains a blob with the resulting images, of the dimensions:

- `BatchLength` is equal to the input `BatchLength`
- `BatchWidth` is equal to the input `BatchWidth`
- `ListSize` is equal to the input `ListSize`
- `Height` is equal to the input `Height / GetStride()`
- `Width` is equal to the input `Width / GetStride()`
- `Depth` is equal to `1`
- `Channels` is equal to the input `Channels * GetStride() * GetStride()`

Each image in the set is split in the same way as the following sample.

Assume we have a 2-channel image `4` by `6`, and `GetStride()` is `2`. The image pixel values are:

```c++
// First channel contents
1,  2,  3,  4,  5,  6,
7,  8,  9,  10, 11, 12,
13, 14, 15, 16, 17, 18,
19, 20, 21, 22, 23, 24,

// Second channel contents
25, 26, 27, 28, 29, 30,
31, 32, 33, 34, 35, 36,
37, 38, 39, 40, 41, 42,
43, 44, 45, 46, 47, 48
```

This image will be transformed into a 8-channel image `2` by `3`, with the contents:

```c++
// First channel contents
1,  3,  5,
7,  9,  11,

// Second channel contents
25, 27, 29,
31, 33, 35,

// Third channel contents
13, 15, 17,
19, 21, 23,

// Fourth channel contents
37, 39, 41,
43, 45, 47

// Fifth channel contents
2,  4,  6,
8,  10, 12,

// Sixth channel contents
26, 28, 30,
32, 34, 36,

// Seventh channel contents
14, 16, 18,
20, 22, 24,

// Eighth channel contents
38, 40, 42,
44, 46, 48
```
