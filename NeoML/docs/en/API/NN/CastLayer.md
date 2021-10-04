# CCastLayer Class

<!--  TOC -->

- [CCastLayer Class](#ccastlayer-class)
    - [Settings](#settings)
        - [Output type](#output-type)
    - [Trainable parameters]
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that performs conversion of input blob into output type.

## Settings

### Output type

```c++
void SetOutputType( TBlobType type );
```

Sets type of the output data.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The single input accepts a blob of any size with any data type.

## Outputs

The single output contains a blob of the same size with `GetOutputType()` type.
