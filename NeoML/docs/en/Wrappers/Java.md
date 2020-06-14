# Java Wrapper

<!-- TOC -->

- [Java Wrapper](#java-wrapper)
	- [Set up the math engine](#set-up-the-math-engine)
	- [Create and read blobs](#create-and-read-blobs)
	- [Load and run the network](#load-and-run-the-network)
	- [Code sample](#code-sample)

<!-- /TOC -->

The Java wrapper for **NeoML** library lets you run an already trained neural network on an Android device. 

The wrapper provides only the minimum API necessary to use the network.

## Set up the math engine
	
To create blob and network objects, you need to set the math engine that will be used for calculations. The math engine is represented by the **`NeoMathEngine`** class.

Use its `CreateCpuMathEngine` or `CreateGpuMathEngine` methods to create a new math engine. 

Call the `close` method to release all resources after use.

```kotlin
/// A single-thread CPU math engine
val neoMathEngine: NeoMathEngine = NeoMathEngine.CreateCpuMathEngine( 1 )
```

## Create and read blobs

A blob is represented by the **`NeoBlob`** class. 

Use its `CreateDnnBlob` method to create a new blob and the `SetData` method to fill its contents from memory.

The network output will also be returned as a blob. To retrieve the data from it use the `GetData` method.

Call the `close` method to release all resources after use.

```kotlin
/// Create an input blob for NeoML
val blob: NeoBlob = NeoBlob.CreateDnnBlob(
    neoMathEngine,
    NeoBlob.Type.FLOAT32,
    1, // only 1 image in the input
    1, // process 1 image at a time
    224, 224, 1, // image resolution ( H, W, D )
    3 // RGB components without alpha
)
```

## Load and run the network

The neural network is represented by the **`NeoDnn`** class.

To start using the previously trained network, load it into a memory buffer (a direct buffer is recommended to decrease the overhead on copying operations) and call the `CreateDnn` method.

Set the data blobs that should be passed into the network by several calls to the `SetInputBlob` method. Run the network using the `Run` method. Note that this method performs only the forward pass, calculating the network response on the given inputs. Backpropagation and training is not performed.

After the network run, retrieve the response from the outputs using the `GetOutputBlob`. Do this before running the network again, as on a new run the old output blobs become invalid.

You may run the network as many times as necessary. Call the `close` method to release all resources after use.

```kotlin
// Set the image data input
blob.SetData(buff)
// Initialize the network.
val neoDnn: NeoDnn = NeoDnn.CreateDnn(neoMathEngine, model)
// Fill the blob with image data
neoDnn.SetInputBlob(0, blob)
// Run the network
neoDnn.Run()

// Get the result data
val outputBlob = neoDnn.GetOutputBlob(0)
val result = outputBlob.GetData().asFloatBuffer()
```

If you have an ONNX file with a model trained by another framework, load it using the `CreateDnnFromOnnx` method. Then set the inputs and run it in the same way.

```kotlin
NeoDnn = NeoDnn.CreateDnnFromOnnx(neoMathEngine, model)
```

## Code sample

For a hands-on example of working with the Java wrapper, check out this sample in Kotlin: [samples/kotlin/classifier](../../../samples/kotlin/classifier). It uses a pretrained neural network to sort a set of images into pictures of documents (with text on them) and pictures that contain no text.