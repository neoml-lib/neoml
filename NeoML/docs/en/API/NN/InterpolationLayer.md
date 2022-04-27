# CInterpolationLayer Class

<!-- TOC -->

- [CInterpolationLayer Class](#cinterpolationlayer-class)
    - [Settings](#settings)
        - [Scale](#scale)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that scales some axes of the blob and fills the new elements with interpolated values.

At this moment only linear interpolation is supported.

## Settings

### Scale

```c++
void SetScale( TBlobDim dim, int scale );
```

Sets the scale of the given blob dimension. `scale` must be more or equal to `1`. `1` by default for all of the dimensions.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

The single input accepts a blob of any size.

## Outputs

The single output contains a blob of the following size:

for each of the blob dimensions, the output size is equal to input size multiplied by its `GetScale`.
