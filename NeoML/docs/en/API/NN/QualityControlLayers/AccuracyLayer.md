# CAccuracyLayer Class

<!-- TOC -->

- [CAccuracyLayer Class](#caccuracylayer-class)
    - [Settings](#settings)
        - [Resetting the data after each run](#resetting-the-data-after-each-run)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

This class implements a layer that calculates classification *accuracy*, that is, the proportion of objects classified correctly in the set.

## Settings

### Resetting the data after each run

```c++
void SetReset( bool value );
```

Specifies if the data should be reset after each network run. By default, the reset is turned **on**.

If you turn off this setting, the total accuracy since the last reset will be calculated.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer has two inputs. The first input accepts a blob with the network response, of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to the number of objects that were classified.
- `Height`, `Width`, and `Depth` are equal to `1`.
- `Channels` is equal to `1` for binary classification and to the number of classes if there are more than 2.

The second input should contain a blob of the same dimensions with the correct class labels:

- If first input `Channels` is equal to `1`, the labels for the binary classification should contain `1` for one class and `-1` for the other.
- If `Channels` is greater than `1`, the labels should contain `1` for correct class and `0` for the others.

## Outputs

The single output returns a blob with only one element, which contains the proportion of correctly classified objects among all objects.

If you have set `SetReset()` to `false`, the layer will accumulate the data for all network runs since the last reset.
