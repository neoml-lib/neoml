# IMathEngine Interface

<!-- TOC -->

- [IMathEngine Interface](#imathengine-interface)
    - [General principles](#general-principles)
        - [Memory management](#memory-management)
            - [CMemoryHandle](#cmemoryhandle)
        - [Method parameters](#method-parameters)
        - [Return values](#return-values)
        - [Synchronizing with GPU](#synchronizing-with-gpu)
        - [Warmup](#warmup)
    - [Create and set up the IMathEngine object](#create-and-set-up-the-imathengine-object)
        - [Exception handling](#exception-handling)
        - [Create the default math engine for CPU](#create-the-default-math-engine-for-cpu)
        - [Create the recommended math engine for GPU](#create-the-recommended-math-engine-for-gpu)
        - [Create a CPU math engine](#create-a-cpu-math-engine)
        - [Create a GPU math engine](#create-a-gpu-math-engine)
        - [GPU math engine manager](#gpu-math-engine-manager)

<!-- /TOC -->

The purpose of the IMathEngine interface is to isolate the algorithms library from the implementation of the low-level platform-dependent operations. The interface provides methods for memory management and calculations. It is used in blob, layer, and neural network objects.

The **NeoML** library supports various processing devices and platform technologies:

Platform | CPU | GPU
----------|-----|-----
Windows | OpenMP + MKL | CUDA
Linux | OpenMP + MKL | -
MacOS | OpenMP + MKL | -
Android | OpenMP + ARM Neon | Vulkan
iOS | ARM Neon | Metal

## General principles

All you need to work with the library is [creating](#create-and-set-up-the-imathengine-object) and, once processing is completed, destroying an IMathEngine object. This section gives the general information about the math engine internals that you will not normally need to access.

### Memory management

The library does not access the memory directly, because it may be allocated on GPU RAM. Because of this, `IMathEngine` manages the memory via special types and functions.

#### CMemoryHandle

`CMemoryHandle` is the base class for all data descriptors. An instance of this type describes a memory block of arbitrary or unknown type. In a way, this class is similar to `void*` data type in C/C++.

Two classes are derived from it: `CFloatHandle` and `CIntHandle`; they describe memory blocks with `float` and `int` data, respectively.

### Method parameters

The math engine processes only vectors of a specific kind. Both the input data and the result should be in vector form.

Any kind of data may be represented as vector: a number is a vector of only one element, a matrix is a vector that contains its data written out row-by-row, and so on for tensors of 3 or more dimensions. This concept allows the math engine to work around the particulars of memory management on different platforms including GPU.

### Return values

In most cases, the math engine methods have no return value; that is, the type of return value is `void` and no non-constant references or pointers are used as "out-parameters." This helps avoid unnecessary CPU-to-GPU synchronization overhead that could significantly impact performance.

### Synchronizing with GPU

Nevertheless there are cases when synchronization is needed: for example, calculations were performed on GPU but the result is required on the "main" system.

See the full list of possible situations when CPU and GPU have to be synchronized:

- memory allocation and release
- reading data from disk
- writing large blocks of data

These cases will require additional synchronization resources, which will probably reduce speed.

### Warmup

Note that the system has to be "warmed up" before actual processing. As many internal objects use lazy initialization, the first run of the math engine may be slower than the subsequent ones.

However, the warmup is guaranteed to be one-off: if you will call the same methods for vectors of the same size many times, the processing will only be slower at first. For example, when training or running a neural network, all required memory buffers will be created on the first run, and the other operations will run faster.

## Create and set up the IMathEngine object

There are several ways to create or get the pointer to the created math engine.

The default math engine will be deleted automatically on unloading the library. All other math engines should be deleted after use, but before that you need to free all memory used by the math engine, deleting all blobs created for this engine (note that blobs may be stored inside layer and network objects, so all those objects should be deleted as well).

### Exception handling

By default, when the exceptional situation occurs `NeoML` functions throw `std::logic_error` or `std::bad_alloc` in case of memory allocation failure.

But this behavior can be changed by setting the exception handler.

```c++
// Exception handler interface
// Use it to change the program's reaction to exceptions
class NEOMATHENGINE_API IMathEngineExceptionHandler {
public:
	virtual ~IMathEngineExceptionHandler();
	// An error during a method call
	// The default action is to throw std::logic_error
	virtual void OnAssert( const char* message, const wchar_t* file, int line, int errorCode ) = 0;

	// Memory cannot be allocated on device
	// The default action is to throw std::bad_alloc
	virtual void OnMemoryError() = 0;
};

// Set exception handler interface for whole programm
// Set this to null to use default exception handler
// Non-default handler must be destroyed by the caller after use
NEOMATHENGINE_API void SetMathEngineExceptionHandler( IMathEngineExceptionHandler* exceptionHandler );

// Get current exception handler interface
// Returns null if use default
NEOMATHENGINE_API IMathEngineExceptionHandler* GetMathEngineExceptionHandler();
```

In order to use non-default exception handling it's recommended to set the exception handler before the creation of math engines.

### Create the default math engine for CPU

```c++
IMathEngine& GetDefaultCpuMathEngine();
```

Returns a math engine working on CPU that uses only one processing thread and has no memory limitations.

This math engine does not need to be deleted after use (the memory and resources will be freed up automatically on unloading the library).

### Create the recommended math engine for GPU

```c++
IMathEngine* GetRecommendedGpuMathEngine( size_t memoryLimit );
```

Creates a math engine working on the recommended GPU. If no GPUs are available `null` will be returned.

#### Parameters

* *memoryLimit* - the memory limitation for the math engine. Set to `0` to use all available memory.

This math engine should be deleted after use.

### Create a CPU math engine

```c++
IMathEngine* CreateCpuMathEngine( int threadCount, size_t memoryLimit );
```

Creates a math engine working on CPU, setting the memory limitation, the number of threads and the custom exception handler.

#### Parameters

* *threadCount* - the maximum number of threads in use.
* *memoryLimit* - the memory limitation for the math engine. Set to `0` to use all available memory.

This math engine should be deleted after use.

### Create a GPU math engine

```c++
IMathEngine* CreateGpuMathEngine( size_t memoryLimit );
```

Creates a math engine working on GPU, setting the memory limitation and the custom exception handler.

#### Parameters

* *memoryLimit* - the memory limitation for the math engine. Set to `0` to use all available memory.

This math engine should be deleted after use.

### GPU math engine manager

Use the GPU manager to get the information about available GPUs and create a math engine working on one of them.

The manager is represented by the interface:

~~~c++
class IGpuMathEngineManager {
public:
	// Get the number of available GPUs
	virtual int GetMathEngineCount() const = 0;

	// Get the information about the GPU with the specified index
	// index can be from 0 to GetMathEngineCount() - 1.
	virtual void GetMathEngineInfo( int index, CMathEngineInfo& info ) const = 0;

	// Create a math engine on the GPU with the specified index
	// index can be from 0 to GetMathEngineCount() - 1.
	// memoryLimit is the memory limitation; if the limit is exceeded IMathEngineExceptionHandler::OnMemoryError() will be thrown
	virtual IMathEngine* CreateMathEngine( int index, size_t memoryLimit ) const = 0;
};
~~~

Create or destroy the manager object:

~~~c++
// Creates a GPU manager
IGpuMathEngineManager* CreateGpuMathEngineManager();

// Destroys the GPU manager
void DestroyGpuMathEngineManager( IGpuMathEngineManager* manager );
~~~

Any math engine created via the manager should be deleted after use.
