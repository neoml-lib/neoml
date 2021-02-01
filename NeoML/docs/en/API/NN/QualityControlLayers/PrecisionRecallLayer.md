# CPrecisionRecallLayer Class

<!-- TOC -->

- [CPrecisionRecallLayer Class](#cprecisionrecalllayer-class)
    - [Settings](#settings)
        - [Resetting the data after each run](#resetting-the-data-after-each-run)
    - [Trainable parameters](#trainable-parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
	    - [Getting the results](#getting-the-results)

<!-- /TOC -->

This class implements a layer that calculates the number of objects classified correctly for either class in a binary classification scenario.

Using these statistics, you can easily calculate the *precision* and *recall* for the trained network.

## Settings

### Resetting the data after each run

```c++
void SetReset( bool value );
```
Specifies if the data should be reset after each network run. By default, the reset is turned **on**.

If you turn off this setting, the total values since the last reset will be calculated.

## Trainable parameters

This layer has no trainable parameters.

## Inputs

The layer has two inputs. The first input accepts a blob with the network response, of the dimensions:

- `BatchLength * BatchWidth * ListSize` is equal to the number of objects that were classified.
- `Height`, `Width`, `Depth`, and `Channels` are equal to `1`.

The second input should contain a blob of the same dimensions with the correct class labels (`1` for one class and `-1` for the other).

## Outputs

The single output contains a blob of the dimensions:

- `Channels` is equal to `4`
- all other dimensions are equal to `1`

The four elements of the blob contain:

1. The number of objects of the `1` class that were classified correctly.
2. The total number of the `1` class objects.
3. The number of objects of the `-1` class that were classified correctly.
4. The total number of the `-1` objects.

If you have set `SetReset()` to `false`, the layer will accumulate the data for all network runs since the last reset.

### Getting the results

```c++
void GetLastResult( CArray<int>& results );
```

Writes the four statistics into an array in the same order as for the [output](#outputs) blob.
