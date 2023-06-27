# NeoOnnx Library

<!-- TOC -->
- [NeoOnnx Library](#neoonnx-library)
    - [API](#api)
        - [Import a network](#import-a-network)
    - [Build](#build)
    - [Implementation](#implementation)
    - [Mobile support](#mobile-support)
<!-- /TOC -->

The **NeoOnnx** library lets you load third-party neural networks serialized in ONNX format.

## API

### Import a network

```c++
#include <NeoOnnx/NeoOnnx.h>

NEOONNX_API void LoadFromOnnx( const char* fileName, const CImportSettings& settings,
    NeoML::CDnn& dnn, CImportedModelInfo& info );
NEOONNX_API void LoadFromOnnx( const void* buffer, int bufferSize, const CImportSettings& settings,
    NeoML::CDnn& dnn, CImportedModelInfo& info );
```

Loads a network from a file or a buffer.

For each network input the `dnn` network will have a `CSourceLayer` with the same name. For each source layer a blob of the size specified in the ONNX model will be allocated. Also input names will be added to the `inputs` array. The inputs with initializers will be ignored and the initializer values will be loaded directly.

For each network output the `dnn` network will have a `CSinkLayer` with the same name. Also output names will be added to the `outputs` array.

Information about model's inputs, outputs and `metadata_props` will be written to the `info`.

## Build

The library will be built automatically together with **NeoML**.

## Implementation

We use the ONNX opset version 9, supporting the main convolutional neural network operations, LSTM, and most activation functions.

## Mobile support

See the methods that load ONNX models in [Objective-C](../en/Wrappers/ObjectiveC.md) and [Java](../en/Wrappers/Java.md) interfaces.
