# CConfusionMatrixLayer Class

<!-- TOC -->

- [CConfusionMatrixLayer Class](#cconfusionmatrixlayer-class)
    - [Settings](#settings)
        - [Resetting the matrix after each run](#resetting-the-matrix-after-each-run)
        - [Manual reset](#manual-reset)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates the `Confusion Matrix` for classification results.

`Confusion Matrix` is a square matrix of the size equal to the number of classes. The columns correspond to the classes determined by the network, the rows - to the actual classes to which the objects belong. Each element contains the number of objects which belong to the `row` class and were classified as the `column` class.

If classification was correct for all objects, the confusion matrix should be diagonal (all non-diagonal elements equal to `0`).

## Settings

### Resetting the matrix after each run

```c++
void SetReset( const bool value );
```

Specifies if the matrix should be reset (filled with zeros) after each network run. By default, the reset is turned **on**.

If you turn off this setting, the matrix will contain the total results since the last reset.

### Manual reset

```c++
void ResetMatrix();
```

Resets all matrix elements to `0`.

## Trainable parameters

The layer has no trainable parameters.

## Inputs

The layer has two inputs. The first input accepts a blob with the network response, of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to the number of objects that were classified.
- `Height`, `Width`, and `Depth` are equal to `1`.
- `Channels` is equal to the number of classes (and should be greater than `1`).

The second input accepts a blob with the correct classes for the objects. Its dimensions should be the same.

## Outputs

The single output returns a blob of the dimensions:

- `BatchLength`, `BatchWidth`, `ListSize`, `Depth`, and `Channels` are equal to `1`.
- `Height` and `Width` are equal to the input `Channels`.

The column number in the matrix means the class to which the network assigned the object; the row number means the correct class.

If you have set `SetReset()` to `false`, the layer will accumulate the data for all network runs until you reset it manually.
