# CCenterLossLayer Class

<!-- TOC -->

- [CCenterLossLayer Class](#ccenterlosslayer-class)
    - [Settings](#settings)
        - [Classes number](#classes-number)
        - [Class centers convergence](#class-centers-convergence)
        - [Loss weight](#loss-weight)
        - [Gradient clipping](#gradient-clipping)
    - [Trainable parameters](#trainable-parameters)
        - [Class centers](#class-centers)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
        - [Getting the value of the loss function](#getting-the-value-of-the-loss-function)

<!-- /TOC -->

This class implements a layer that penalizes large differences between objects of the same class. See the [paper](http://ydwen.github.io/papers/WenECCV16.pdf) for details.

The layer may be used as a loss function.

## Settings

### Classes number

```c++
void SetNumberOfClasses( int numberOfClasses );
```

Sets the number of classes in the model.

### Class centers convergence

```c++
void SetClassCentersConvergenceRate( float classCentersConvergenceRate );
```

Sets the multiplier used for calculating the moving mean of the class centers for each subsequent iteration. This value must be between 0 and 1.

### Loss weight

```c++
void SetLossWeight( float lossWeight );
```

Sets the multiplier for this function gradient during training. The default value is `1`. You may wish to change the default if you are using several loss functions in your network.

### Gradient clipping

```c++
void SetMaxGradientValue( float maxValue );
```

Sets the upper limit for the absolute value of the function gradient. Whenever the gradient exceeds this limit its absolute value will be reduced to `GetMaxGradientValue()`.

## Trainable parameters

### Class centers

```c++
CPtr<const CDnnBlob> GetClassCenters();
```

Retrieves the class centers that were calculated during the operation of the other layers of the network. The resulting blob is a two-dimensional matrix with `GetNumberOfClasses()` rows and `Height * Width * Depth * Channels` columns (calculated from the dimensions of the first input blob).

## Inputs

The layer may have 2 to 3 inputs:

1. The network output for which you are calculating the loss function.
2. The class labels represented by a blob with integer data. Each element in the blob contains the number of the class to which the object with this index in the input belongs. The dimensions of the blob are:

	- `BatchLength`, `BatchWidth`, `ListSize` are equal to the corresponding dimensions of the first input.
	- all other dimensions are equal to `1`.

3. *[Optional]* The objects' weights. This input should have the same `BatchLength`, `BatchWidth`, and `ListSize` dimensions as the first input. `Height`, `Width`, `Depth`, and `Channels` should be equal to `1`.

## Outputs

This layer has no output.

### Getting the value of the loss function

```c++
float GetLastLoss() const;
```

Use this method to get the value of the loss function calculated on the network's last run.
