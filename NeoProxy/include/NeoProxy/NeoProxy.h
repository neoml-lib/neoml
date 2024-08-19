/* Copyright © 2017-2024 ABBYY

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#pragma once

#if __OBJC__ && __cplusplus

#define NEOPROXY_API __attribute__((visibility("default")))

#else

#include <stdbool.h>
#include <NeoProxy/NeoProxyDefs.h>

#endif

// The minimum API for neural network use. Allows you to load and run a network (only forward pass).

#ifdef __cplusplus
extern "C" {
#endif

// Error type
enum TDnnErrorType {
	DET_OK = 0, // no error
	DET_InternalError, // internal error, details in error description
	DET_NoAvailableGPU, // no GPU available
	DET_NoAvailableCPU, // no CPU available
	DET_InvalidParameter, // parameter value is invalid
	DET_RunDnnError, // error when running the network, details in error description
	DET_LoadDnnError // error when loading the network, details in error description
};

// Error description
struct NEOPROXY_API CDnnErrorInfo {
	enum TDnnErrorType Type; // Error type
	char Description[512]; // Error message
};

//---------------------------------------------------------------------------------------------------------------------

// Math engine functions

// The type of processing unit used for the math engine
enum TDnnMathEngineType {
	MET_CPU = 0,
	MET_GPU
};

// The math engine descriptor
struct NEOPROXY_API CDnnMathEngineDesc {
	enum TDnnMathEngineType Type;
};

// Creates a math engine that uses a GPU for calculations
// This math engine should be deleted after use with the help of the DestroyMathEngine() function
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const struct CDnnMathEngineDesc* CreateGPUMathEngine( struct CDnnErrorInfo* errorInfo );

// Creates a math engine that uses a CPU for calculations
// This math engine should be deleted after use with the help of the DestroyMathEngine() function
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const struct CDnnMathEngineDesc* CreateCPUMathEngine( int threadCount, struct CDnnErrorInfo* errorInfo );

// Destroys the math engine
// Only do this after destroying all the blobs and networks created for this math engine
NEOPROXY_API void DestroyMathEngine( const struct CDnnMathEngineDesc* mathEngine );

//---------------------------------------------------------------------------------------------------------------------

// Blob functions

// Type of data in a blob
enum TDnnBlobType {
	DBT_Float = 1,
	DBT_Int = 2
};

// The blob descriptor
struct NEOPROXY_API CDnnBlobDesc {
	const struct CDnnMathEngineDesc* MathEngine; // the math engine for the blob
	enum TDnnBlobType Type; // the type of data in the blob
	int BatchLength; // sequence length
	int BatchWidth; // the number of sequences processed together
	int Height; // object height
	int Width; // object width
	int Depth; // object depth
	int ChannelCount; // the number of channels
	int DataSize; // the total blob size in bytes
};

// Creates a data blob
// The blob should be destroyed after use with the help of the DestroyDnnBlob function
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const struct CDnnBlobDesc* CreateDnnBlob( const struct CDnnMathEngineDesc* mathEngine, enum TDnnBlobType type,
	int batchLength, int batchWidth, int height, int width, int depth, int channelCount, struct CDnnErrorInfo* errorInfo );

// Destroys the blob
NEOPROXY_API void DestroyDnnBlob( const struct CDnnBlobDesc* blob );

// Fills the blob with data from a buffer
// If successful, returns true; if failed, returns false and fills the errorInfo parameter with the error description
NEOPROXY_API bool CopyToBlob( const struct CDnnBlobDesc* blob, const void* buffer, struct CDnnErrorInfo* errorInfo );

// Copies the blob data into a buffer
// If successful, returns true; if failed, returns false and fills the errorInfo parameter with the error description
NEOPROXY_API bool CopyFromBlob( void* buffer, const struct CDnnBlobDesc* blob, struct CDnnErrorInfo* errorInfo );

//---------------------------------------------------------------------------------------------------------------------

// Neural network functions

// The neural network descriptor
struct NEOPROXY_API CDnnDesc {
	const struct CDnnMathEngineDesc* MathEngine; // the math engine for the network
	int InputCount; // the number of network inputs
	int OutputCount; // the number of network outputs
};

// Loads the network from a file. It should implement the NeoML archive and contain the serialized network
// The network should be destroyed after use with the help of the DestroyDnn function
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const struct CDnnDesc* CreateDnnFromFile( const struct CDnnMathEngineDesc* mathEngine, const char* fileName, struct CDnnErrorInfo* errorInfo );

// Loads the network from a memory buffer
// The network should be destroyed after use with the help of the DestroyDnn function
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const struct CDnnDesc* CreateDnnFromBuffer( const struct CDnnMathEngineDesc* mathEngine, const void* buffer, int bufferSize, struct CDnnErrorInfo* errorInfo );

// Loads the network from an onnx file. It should implement the NeoML archive and contain the serialized network
// The network should be destroyed after use with the help of the DestroyDnn function
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const struct CDnnDesc* CreateDnnFromOnnxFile( const struct CDnnMathEngineDesc* mathEngine, const char* fileName, struct CDnnErrorInfo* errorInfo );

// Loads the network from a memory buffer with onnx data in it
// The network should be destroyed after use with the help of the DestroyDnn function
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const struct CDnnDesc* CreateDnnFromOnnxBuffer( const struct CDnnMathEngineDesc* mathEngine, const void* buffer, int bufferSize, struct CDnnErrorInfo* errorInfo );

// Destroys the network
NEOPROXY_API void DestroyDnn( const struct CDnnDesc* dnn );

// Retrieves the name of a network input
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const char* GetInputName( const struct CDnnDesc* dnn, int index, struct CDnnErrorInfo* errorInfo );

// Sets the input data blob before running the network
// If an error occurs its description will be written into the errorInfo parameter and the function will return false
NEOPROXY_API bool SetInputBlob( const struct CDnnDesc* dnn, int index, const struct CDnnBlobDesc* blob, struct CDnnErrorInfo* errorInfo );

// Run the network (perform a forward pass)
// If an error occurs its description will be written into the errorInfo parameter and the function will return false
NEOPROXY_API bool DnnRunOnce( const struct CDnnDesc* dnn, struct CDnnErrorInfo* errorInfo );

// Retrieves the name of a network output
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const char* GetOutputName( const struct CDnnDesc* dnn, int index, struct CDnnErrorInfo* errorInfo );

// Retrieves the output blob
// The blob should be destroyed after use with the help of the DestroyDnnBlob function
// If an error occurs its description will be written into the errorInfo parameter and the function will return 0
NEOPROXY_API const struct CDnnBlobDesc* GetOutputBlob( const struct CDnnDesc* dnn, int index, struct CDnnErrorInfo* errorInfo );

#ifdef __cplusplus
} // extern "C"
#endif
