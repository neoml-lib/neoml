# CSplitDepthLayer Class

<!-- TOC -->

- [CSplitDepthLayer Class](#csplitdepthlayer-class)
    - [Settings](#settings)
        - [Outputs size](#outputs-size)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that splits a single input blob into several output blobs along the `Depth` coordinate.


## Settings

### Output size

```c++
void SetOutputCounts(const CArray<int>& outputCounts);
```

Sets the values of the `Depth` dimension for the output blobs. See [below](#outputs) for more details on the size and number of the outputs.


```c++
void SetOutputCounts2(int count0);
void SetOutputCounts3(int count0, int count1);
void SetOutputCounts4(int count0, int count1, int count2);
```

These auxiliary methods set the values of the `Depth` dimension for the outputs blobs, in the cases where you want 2, 3, or 4 outputs. Equivalent to calling the `SetOutputCounts(const CArray<int>&)` with the array of `1`, `2`, or `3` elements. See [below](#outputs) for more details on the size and number of the outputs.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

There is a single input which accepts a blob of any size. The only limitation is that its `Depth` dimension must be greater or equal to the sum of the elements of the `GetOutputCounts()` array.

## Outputs

The layer has at least `GetOutputCounts().Size()` outputs. Each of the outputs contains a blob of the dimensions:

- `BatchLength`, `BatchWidth`, `ListSize`, `Height`, `Width`, `Channels` are equal to the corresponding dimensions of the input.
- `Depth` is equal to the corresponding element of the `GetOutputCount()` array; for example, the first output `Depth` is equal to `GetOutputCount()[0]` and so on.

However, if the `Depth` dimension of the input is **greater** than the sum of the `GetOutputCounts()` elements, the layer will have one more output of the dimensions:

- `BatchLength`, `BatchWidth`, `ListSize`, `Height`, `Width`, `Channels` are equal to the corresponding dimensions of the input.
- `Depth` is equal to the difference between the input `Depth` and the sum of the `GetOutputCount()` elements.
