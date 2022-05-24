# CInterpolationLayer Class

<!-- TOC -->

- [CInterpolationLayer Class](#cinterpolationlayer-class)
    - [Settings](#settings)
        - [Dimension sizes](#dimension-sizes)
        - [Coordinate system](#coordinate-system)
        - [Coordinate rounding](#coordinate-rounding)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that changes the sizes of some axes of the blob and fills the new elements with interpolated values.

At this moment only linear interpolation is supported.

## Settings

### Dimension sizes

```c++
void SetRule( TBlobDim dim, const CRule& rule );
```

Sets the rule to be used when calculating the size of `dim` blob dimension.

It's recommended to use one of these functions to create the `CRule`:
- `CInterpolationLayer::CRule::Resize( int newSize )` to set the `dim` of the output to `newSize`
- `CInterpolationLayer::CRule::Scale( float scale )` to multiply the `dim` by `scale` (relative to the input blob).

By default layer doesn't change any of the dimensions.

### Coordinate system

```c++
void SetCoords( TInterpolationCoords newCoords );
```

Sets the way layer calculates the coordinates of the new elements `xNew` relative to the elements of the input `xOld`.

Supported values:
- `TInterpolationCoords::HalfPixel` - `xOld = ( xNew + 0.5 ) / scale - 0.5`
- `TInterpolationCoords::PytorchHalfPixel` - `xOld = ( newSize > 1 ) ? ( xNew + 0.5 ) / scale - 0.5 : 0`
- `TInterpolationCoords::AlignCorners` - `xOld = xNew * ( oldSize - 1 ) / ( newSize - 1 )`
- `TInterpolationCoords::Asymmetric` - `xOld = xNew / scale`
where:
- `oldSize` and `newSize` are the sizes of the dimension before and after the transformation
- `scale` - coefficient used to multiply input dimension (`newSize / oldSize` in case of fixed output size)

By default `TInterpolationCoords::Asymmetric` is used.

### Coordinate rounding

```c++
void SetRound( TInterpolationRound newRound );
```

Sets how the coordinates `xOld` are rounded. If used then every element of the output will be one of the elements of the input (no interpolated values).

Supported values:
- `TInterpolationRound::None` - don't round, calculate interpolated values
- `TInterpolationRound::RoundPreferFloor` - less or equal to the half values are rounded down
- `TInterpolationRound::RoundPreferCeil` - less than half values are rounded down
- `TInterpolationRound::Floor` - zeroing non-integer part
- `TInterpolationRound::Ceil` - if non-integer part is present, it's rounded up.

By default rounding is not used (`TInterpolationRound::None`).

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

The single input accepts a blob of any size.

## Outputs

The single output contains a blob where each `dim` size is calculated based on `GetRule( dim )`.