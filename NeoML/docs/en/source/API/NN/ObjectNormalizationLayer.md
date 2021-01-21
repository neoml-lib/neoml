# CObjectNormalizationLayer Class

<!-- TOC -->

- [CObjectNormalizationLayer Class](#cobjectnormalizationlayer-class)
    - [Settings](#settings)
        - [Epsilon](#epsilon)
    - [Trainable parameters](#trainable-parameters)
        - [Scale](#scale)
        - [Bias](#bias)
    - [Inputs](#inputs)
    - [Outputs](#outputs)

<!-- /TOC -->

The class implements a layer that performs object normalization using the following formula:

```c++
objectNorm(x)[i][j] = ((x[i][j] - mean[i]) / sqrt(var[i] + epsilon)) * scale[j] + bias[j]
```

where:

- `scale` and `bias` are the trainable parameters
- `mean` and `var` are mean and varian of the obects in a batch.

## Settings

### Epsilon

```c++
void SetEpsilon( float newEpsilon );
```

Sets `epsilon` which is added to the variance in order to avoid division by zero.

## Trainable parameters

### Scale

```c++
CPtr<CDnnBlob> GetScale() const;
```

Gets the scale vector. It is a blob of any shape and of total size equal `Height * Width * Depth * Channels` of the input.

### Bias

```c++
CPtr<CDnnBlob> GetBias() const;
```

Gets the bias vector. It is a blob of any shape and of total size equal `Height * Width * Depth * Channels` of the input.

## Inputs

The single input accepts a blob containing `BatchLength * BatchWidth * ListSize` objects of size `Height * Width * Depth * Channels`.

## Outputs

The single output contains a blob with the results, of the same size as the input blob.
