# Objective-C Wrapper

<!-- TOC -->

- [Objective-C Wrapper](#objective-c-wrapper)
	- [Set up the math engine](#set-up-the-math-engine)
	- [Create and read blobs](#create-and-read-blobs)
	- [Load and run the network](#load-and-run-the-network)
	- [Code sample](#code-sample)

<!-- /TOC -->

The Objective-C wrapper for **NeoML** library lets you run an already trained neural network on an iOS device. 

The wrapper provides only the minimum API necessary to use the network.

## Set up the math engine

To create blob and network objects, you need to set the math engine that will be used for calculations. The math engine is represented by the **`NeoMathEngine`** class.

Use its `createCpuMathEngine` or `createGpuMathEngine` methods to create a new math engine.

```swift
/// A single-thread CPU math engine
let mathEngine: NeoMathEngine = try NeoMathEngine.createCPUMathEngine( 1 )
```

## Create and read blobs

A blob is represented by the **`NeoBlob`** class. 

Use its `createDnnBlob` method to create a new blob and the `setData` method to fill its contents from memory.

The network output will also be returned as a blob. To retrieve the data from it use the `getData` method.

```swift
// Create an input blob for NeoML
let inputBlob = try NeoBlob.createDnnBlob(mathEngine,
                                          blobType: .float32,
                                          batchLength: 1, // only 1 image in the input
                                          batchWidth: 1, // process 1 image at a time
                                          height: 224, // image height
                                          width: 224,  // image width
                                          depth: 1, // image depth
                                          channelCount: 3) // RGB components without alpha
// Set the image data input
try inputBlob.setData(data)
```

## Load and run the network

The neural network is represented by the **`NeoDnn`** class.

To start using the previously trained network, load it into a memory buffer and call the `createDnn` method.

Set the data blobs that should be passed into the network by several calls to the `setInputBlob` method. Run the network using the `run` method. Note that this method performs only the forward pass, calculating the network response on the given inputs. Backpropagation and training is not performed.

After the network run, retrieve the response from the outputs using the `getOutputBlob`. Do this before running the network again, as on a new run the old output blobs become invalid.

```swift
// Initialize the network
let dnn = try NeoDnn.createDnn(mathEngine, data: networkFileData)
// Fill the blob with image data
try dnn.setInputBlob(0, blob: inputBlob)
// Run the network
try dnn.run()
// Get the result data
let result = try dnn.getOutputBlob(0)
let outputData = try result.getData()
```

If you have an ONNX file with a model trained by another framework, load it using the `createDnn(fromOnnx:` method. Then set the inputs and run it in the same way.

```swift
let dnn = try NeoDnn.createDnn(fromOnnx: mathEngine, data: onnxFileData)
```

## Code sample

For a hands-on example of working with the Objective-C wrapper, check out this sample in Swift: [samples/swift/classifier](../../../samples/swift/classifier). It uses a pretrained neural network to sort a set of images into pictures of documents (with text on them) and pictures that contain no text.