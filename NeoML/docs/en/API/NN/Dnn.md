# CDnn Class

<!-- TOC -->

- [CDnn Class](#cdnn-class)
    - [Constructor](#constructor)
    - [Operations with layers](#operations-with-layers)
        - [Adding a layer](#adding-a-layer)
        - [Checking if the layer is present](#checking-if-the-layer-is-present)
        - [Accessing a layer](#accessing-a-layer)
        - [Deleting a layer](#deleting-a-layer)
        - [Getting the list of layers](#getting-the-list-of-layers)
        - [Deleting all layers](#deleting-all-layers)
    - [Training the network](#training-the-network)
        - [Weights initialization](#weights-initialization)
        - [The optimizer](#the-optimizer)
        - [Training iteration](#training-iteration)
    - [Running the network](#running-the-network)
    - [Serialization](#serialization)
    - [Logging](#logging)

<!-- /TOC -->

This class implements a neural network. A neural network is a directed graph consisting of layers that perform calculations on data blobs. The graph starts with *source* layers and ends with *sink* layers.

## Constructor

```c++
CDnn::CDnn( IMathEngine& mathEngine, CRandom& random )
```

Specify the [math engine](MathEngine.md) and the random numbers generator in the constructor. Note that the network and all its layers have to use the same math engine.

## Operations with layers

### Adding a layer

```c++
void AddLayer( CBaseLayer& layer );
```

Adds a layer to the neural network. An error will occur if the layer is already in a network, or the network already has a layer with the same name.

### Checking if the layer is present

```c++
virtual bool HasLayer( const char* name ) const;
```

Checks if a layer with the specified name is present in the network.

### Accessing a layer

```c++
virtual CPtr<CBaseLayer> GetLayer( const char* name );
virtual CPtr<const CBaseLayer> GetLayer( const char* name ) const;
```

Retrieves a pointer to the layer with the specified name. If no such layer exists, an error will occur.

### Deleting a layer

```c++
void DeleteLayer( const char* name );
void DeleteLayer( CBaseLayer& layer );
```

Removes the specified layer from the network. If it is not connected to the network, an error will occur.

### Getting the list of layers

```c++
virtual void GetLayerList( CArray<const char*>& layerList ) const;
```

Retrieves the list of names of all layers in the network.

### Deleting all layers

```c++
void CDnn::DeleteAllLayers();
```

Removes all layers from the network.

## Training the network

### Weights initialization

```c++
CPtr<CDnnInitializer> GetInitializer();
void SetInitializer( const CPtr<CDnnInitializer>& initializer );
```

The class that should be used to initialize the layer weights before starting training. Xavier initialization is used by default.

### The optimizer

```c++
CDnnSolver* GetSolver();
const CDnnSolver* GetSolver() const;
void SetSolver( CDnnSolver* solver );
```

The class that implements optimization of the layers' trainable parameters.

### Training iteration

```c++
void RunAndLearnOnce();
```

Runs one iteration with training. After this method call you may extract the data from the [sink layers](IOLayers/SinkLayer.md).

## Running the network

```c++
void RunOnce();
```

Runs the network without training. After this method call you may extract the data from the [sink layers](IOLayers/SinkLayer.md).

## Serialization

```c++
void Serialize( CArchive& archive );
```

Serializes the network. If the archive is open for writing, the network will be written into the archive. If it is open for reading, the network layers will be deleted and the new network will be read from the archive.

```c++
void SerializeCheckpoint( CArchive& archive );
```

Serializes the network and additional data, required to resume training from this point (gradient history etc.).

When loading it creates a new optimizer, which can be retrieved by `CDnn::GetSolver` method.

## Logging

```c++
CTextStream* GetLog();
void SetLog( CTextStream* newLog );
```

Retrieves and sets the text stream used to log the network operation.

```c++
int GetLogFrequency() const;
void SetLogFrequency( int logFrequency );
```

Retrieves and sets the logging frequency. By default, every 100th iteration of `RunOnce` or `RunAndLearnOnce` will be logged.
