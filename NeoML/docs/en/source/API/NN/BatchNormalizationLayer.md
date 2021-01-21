# CBatchNormalizationLayer Class

<!-- TOC -->

- [CBatchNormalizationLayer Class](#cbatchnormalizationlayer-class)
    - [Settings](#settings)
        - [Channelwise mode](#channelwise-mode)
        - [Convergence rate](#convergence-rate)
    - [Trainable parameters](#trainable-parameters)
        - [Final values](#final-values)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that performs batch normalization using the following formula:

```c++
bn(x)[i][j] = ((x[i][j] - mean[j]) / sqrt(var[j])) * gamma[j] + beta[j]
```

where:

- `gamma` and `beta` are the trainable parameters
- `mean` and `var` depend on whether the layer is being trained:
	- If the layer is being trained, `mean[j]` and `var[j]` are the mean value and the variance of `x` data with `j` coordinate across all `i`.
	- If the layer is not being trained, `mean[j]` and `var[j]` are the exponential moving mean and the unbiased variance estimate calculated during training.

## Settings

### Channelwise mode

```c++
void SetChannelBased( bool isChannelBased );
```

Turns on and off channel-based statistics. 

If this mode is **on**, `mean`, `var`, `gamma`, and `beta` in the formula will be vectors of the input `Channels` length. The `i` coordinate will iterate over all values from `0` to `BatchLength * BatchWidth * ListSize * Height * Width * Depth - 1`.

If this mode is **off**, the `mean`, `var`, `gamma`, and `beta` vectors will have the `Height * Width * Depth * Channels` length. The `i` coordinate will iterate over all values from `0` to `BatchLength * BatchWidth * ListSize - 1`.

By default the channelwise mode is **on**.

### Convergence rate

```c++
SetSlowConvergenceRate( float rate );
```

Sets the coefficient for calculating the exponential moving mean and variance.

## Trainable parameters

### Final values

```c++
CPtr<CDnnBlob> GetFinalParams();
```

Gets the final values of the parameters. They are returned as a blob of the dimensions:

- `BatchLength` is equal to `1`
- `BatchWidth` is equal to `2`
- `ListSize` is equal to `1`
- `Height` is equal to `1` when `IsChannelBased()`, the input `Height` otherwise
- `Width` is equal to `1` when `IsChannelBased()`, the input `Width` otherwise
- `Depth` is equal to `1` when `IsChannelBased()`, the input `Depth` otherwise
- `Channels` is equal to `1`

The first object of the blob (`BatchWidth` coordinate is equal to `0`) contains the coefficients `gamma[j] / sqrt(var[j])`.

The second object of the blob (`BatchWidth` coordinate is equal to `1`) contains the terms `beta[j] - mean[j] * gamma[j] / sqrt(var[j])`.

The batch normalization formula can then be rewritten as `bn(x)[i][j] = x[i][j] * finalParams[0][j] + finalParams[1][j]`.

## Inputs

The single input accepts a blob of any size.

## Outputs

The single output contains a blob with the results of batch normalization.
