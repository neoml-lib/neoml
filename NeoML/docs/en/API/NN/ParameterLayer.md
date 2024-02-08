# CParameterLayer Class

<!-- TOC -->

- [CParameterLayer Class](#CParameterLayer-class)
    - [Settings](#settings)
        - [Setting blob](#initialization-weight-blob )
    - [Trainable parameters](#trainable-parameters)
        - [Weight blob](#weight-blob)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

Class implements layer with blob with trainable parameters.

## Settings

### Initialization weight blob

```c++
void SetBlob(CDnnBlob* _blob);
```
Setting blob of trainable parameters.

## Trainable parameters

### Weight blob

```c++
const CPtr<CDnnBlob>& GetBlob() const;
```
The blob with trainable weights.

## Inputs

The layer has no inputs.

## Outputs

The output is blob of weights (with same dimensions as the initiated blob had).
