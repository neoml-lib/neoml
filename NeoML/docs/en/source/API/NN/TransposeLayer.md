# CTransposeLayer Class

<!-- TOC -->

- [CTransposeLayer Class](#ctransposelayer-class)
    - [Settings](#settings)
        - [Dimensions to switch](#dimensions-to-switch)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that switches two of the blob dimensions, moving the data inside accordingly.

## Settings

### Dimensions to switch

```c++
void SetTransposedDimensions(TBlobDim d1, TBlobDim d2);
```

Sets the two dimensions that should change places.

## Inputs

The single input accepts a blob of any size.

## Outputs

The single output contains a blob in which the `GetTransposedDimensions()` dimensions are switched together with all the data.
