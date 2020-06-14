# CTransformLayer Class

<!-- TOC -->

- [CTransformLayer Class](#ctransformlayer-class)
    - [Settings](#settings)
        - [Changing the blob dimensions](#changing-the-blob-dimensions)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that changes the input blob dimensions without moving any of the data. The total number of elements in the blob stays the same, and therefore the product of all dimensions should not be changed by the transformation.

## Settings

### Changing the blob dimensions

```c++
// Modes of dimension change
enum TOperation {
    // Set this dimension so that the total size stays the same
    // Only one of the dimensions may have this mode
    O_Remainder,
    // Set this dimension to Parameter value
    O_SetSize,
    // Multiply this dimension by Parameter value
    O_Multiply,
    // Divide this dimension by Parameter value
    O_Divide
};

// The rule of dimension change
struct NEOML_API CDimensionRule {
    // The mode of dimension change
    TOperation Operation;
    // The numerical parameter to be used
    int Parameter;

    CDimensionRule();
    CDimensionRule( TOperation op, int param );

    bool operator==( const CDimensionRule& other ) const;

    // Applies the transformation set by the rule
    int Transform( int input ) const;
};

void SetDimensionRule( TBlobDim dim, const CDimensionRule& rule );
void SetDimensionRule( TBlobDim dim, TOperation op, int param );
```

Sets the change that will be applied to the specified blob dimension. If you do not set any changes for one of the dimensions, it will retain its size.

Only one of the blob dimensions may have the `O_Remainder` mode.

## Trainable parameters

There are no trainable parameters for this layer.

## Inputs

The single input accepts a blob of any size with data of any type.

## Outputs

The single output contains a blob of the dimensions determined by the specified rules:

- The dimensions with `O_SetSize` mode will be equal to the specified `Parameter`.
- The dimensions with `O_Multiply` mode will be `Parameter` times larger.
- The dimensions with `O_Divide` mode will be `Parameter` smaller.
- The dimension with `O_Remainder` mode will be such that the total size of the input and the output are the same.
