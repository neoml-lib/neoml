# CBaseLayer Class

<!-- TOC -->

- [CBaseLayer Class](#cbaselayer-class)
    - [Constructor](#constructor)
    - [The layer name](#the-layer-name)
    - [The network](#the-network)
    - [Connecting to other layers](#connecting-to-other-layers)
    - [The number of inputs](#the-number-of-inputs)
    - [Input information](#input-information)
    - [Manage learning](#manage-learning)
    - [Learning rate multiplier](#learning-rate-multiplier)
    - [Regularization factor](#regularization-factor)

<!-- /TOC -->

This is the base class for a network layer. It contains the interface for interacting with the network ([`CDnn`](#Dnn.md)) and the other layers.

All classes implementing the [network layers](README.MD#the-layers) inherit from this class.

## Constructor

```c++
CBaseLayer( IMathEngine& mathEngine, const char* name, bool isLearnable );
```

Called from the layers' constructors. 

Creates a new layer. You need to pass the reference to the math engine used for calculations, and it should be the same for all the layers of one network. You may also specify the layer's name and if it has trainable weights.

## The layer name

```c++
const char* GetName() const;
void SetName( const char* name );
```

Sets the layer's name. You can only change it for the layers not currently connected to a network.

## The network

```c++
const CDnn* GetDnn() const;
CDnn* GetDnn();
```

Retrieves the pointer to the network to which the layer is connected. Returns `0` if the layer is not in a network.

## Connecting to other layers

```c++
void Connect( int inputNumber, const char* input, int outputNumber = 0 );
void Connect( int inputNumber, const CBaseLayer& layer, int outputNumber = 0 );
void Connect( const char* input );
void Connect( const CBaseLayer& layer );
```

Connects the `inputNumber` input of this layer to the `outputNumber` output of the `layer` layer (or the layer called `input`).

## The number of inputs

```c++
int GetInputCount() const;
```

## Input information

```c++
const char* GetInputName(int number) const;

int GetInputOutputNumber(int number) const;
```

Gets the input description.

## Manage learning

```c++
void DisableLearning();
void EnableLearning();
bool IsLearningEnabled() const;
```

If you turn learning off the layer will not be trained on the call to `CDnn::RunAndLearnOnce`.

## Learning rate multiplier

```c++
float GetBaseLearningRate() const;
void SetBaseLearningRate( float rate );
```

The base learning rate for the layer. It will be multiplied by the `learningRate` from the network optimizer. This setting may be used to change one layer's learning rate relative to the others.

## Regularization factor

```c++
float GetBaseL1RegularizationMult() const;
void SetBaseL1RegularizationMult(float mult);
float GetBaseL2RegularizationMult() const;
void SetBaseL2RegularizationMult( float mult );
```

The base regularization factors. They will be multiplied by the corresponding factors from the network optimizer.
