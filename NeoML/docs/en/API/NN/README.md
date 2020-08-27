# Neural Networks

<!-- TOC -->

- [Neural Networks](#neural-networks)
    - [Choose the math engine](#choose-the-math-engine)
    - [Data blobs](#data-blobs)
    - [General principles](#general-principles)
        - [The layer concept](#the-layer-concept)
        - [CDnn class for the network](#cdnn-class-for-the-network)
    - [Training the network](#training-the-network)
        - [Weights initialization](#weights-initialization)
        - [Optimizers](#optimizers)
        - [Training iteration](#training-iteration)
        - [Running the network](#running-the-network)
    - [Serialization](#serialization)
        - [Sample code for saving the network](#sample-code-for-saving-the-network)
    - [Using the network](#using-the-network)
    - [The layers](#the-layers)

<!-- /TOC -->

## Choose the math engine

Before you start your work with neural networks, choose the device to be used for calculations. This can be a CPU or a GPU. Create a [math engine](MathEngine.md) for the required device and pass the reference to it when creating the network and the layers.

## Data blobs

All data used in the network operation (inputs, outputs, trainable parameters) is stored in [blobs](DnnBlob.md). A *blob* is a 7-dimensional array, and each of its dimensions has a specific meaning:

- `BatchLength` is a "time" axis, used to denote data sequences; it is mainly used in recurrent networks
- `BatchWidth` corresponds to the batch, used to pass several independent objects together
- `ListSize` is the dimensions for the objects that are connected (for example, pixels out of one image) but do not form a sequence
- `Height` is the height of a matrix or an image
- `Width` is the width of a matrix or an image
- `Depth` is the width of a 3-dimensional image
- `Channels` corresponds to channels for multi-channel image formats and is also used to work with one-dimensional vectors.

The blobs may contain one of the two types of data: float (`CT_Float`) and integer (`CT_Int`). Both data types are 32-bit.

If the data type is not specified directly anywhere in this documentation, that means `float` is used.

## General principles

### The layer concept

A [layer](BaseLayer.md) is an element of the network that performs some operation: anything from the input data reshape or a simple math function calculation, up to convolution or LSTM ([Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)).

If the operation needs input data, it will be taken from the layer input. Each layer input contains one data blob, and if several blobs are needed, the layer will have several inputs. Each layer input should be [connected](BaseLayer.md#connecting-to-other-layers) to another layer's output.

If the operation returns results that should be used by other layers, they will be passed to the layer outputs. Each layer output contains one data blob, so depending on the operation it performs the layer may have several outputs. Several other layer inputs may be connected to the same output, but you may not leave an output unconnected to any inputs.

In addition, the layer may have settings specified by the user before starting calculations, and trainable parameters that are optimized during network training.

The layers also have [names](BaseLayer.md#the-layer-name) that can be used to find a layer in the network. The name should be set at layer creation or before adding it to the network.

See [below](#the-layers) for the full list of available layers with links to the detailed descriptions.

### CDnn class for the network

The neural network is implemented by a [CDnn](Dnn.md) class. A neural network is a directed graph with the vertices corresponding to layers and the arcs corresponding to the connections along which the data is passed from one layer's output to another's input.

Each layer should be [added](Dnn.md#adding-a-layer) to the network after you assign a unique [name](BaseLayer.md#the-layer-name) to it. A layer may not be connected to several networks at once.

[Source](IOLayers/SourceLayer.md) layers are used to pass the data into the network. A source layer has no inputs and passes the data blob specified by the user to its only output.

[Sink](IOLayers/SinkLayer.md) layers with no outputs are used to retrieve the result of the network operation. They provide a function that returns the blob with data.

After all the layers are added and connected the network may be set up for training.

## Training the network

To train the network you will need:

* a layer (or several layers) that would calculate the [loss function](LossLayers/README.md) to be optimized
* additional source layers that contain the correct labels for input data and the object weights
* the initializer that would be used to assign the values to the weights before starting to optimize them
* the optimizer mechanism that will be used for training

### Weights initialization

Before the first training iteration the layers' weights (trainable parameters) are initialized using the `CDnnInitializer` object. There are two implementations for it:

- `CDnnUniformInitializer` generates the weights using a uniform distribution over a segment from `GetLowerBound` to `GetUpperBound`.
- `CDnnXavierInitializer` generates the weights using the normal distribution `N(0, 1/n)` where `n` is the input size.

To select the preferred initializer, create an instance of one of these classes and pass it to the network using the [`CDnn::SetInitializer`](Dnn.md#weights-initialization) method. The default initialization methods is `Xavier`.

The initializer is the same for all the network trainable weights, except for the free term vectors that are initialized with zeros.

### Optimizers

The optimizer sets the rules to update the weights during training. It is represented by the `CDnnSolver` that has 4 implementations:

- `CDnnSimpleGradientSolver` - [gradient descent with momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum)
- `CDnnAdaptiveGradientSolver` - gradient descent with adaptive momentum ([Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam))
- `CDnnNesterovGradientSolver` - Adam with Nesterov momentum ([Nadam](http://cs229.stanford.edu/proj2015/054_report.pdf))
- `CDnnLambGradientSolver` - [LAMB](https://arxiv.org/pdf/1904.00962.pdf)

To select the preferred optimizer, create an instance of one of these classes and pass it to the network using the [`CDnn::SetSolver`](Dnn.md#the-optimizer) method.

The additional settings for the optimizer are:

- learning rate (`CDnnSolver::SetLearningRate`)
- regularization factors (`CDnnSolver::SetL2Regularization` and `CDnnSolver::SetL1Regularization`)

### Training iteration

After the initializer and the optimizer have been set, you may start the learning process. To do that, set the input data blobs for all source layers and call the `CDnn::RunAndLearnOnce` method.

The method call will perform three internal operations:

1. `Reshape` - calculates the size and allocates memory for the output blobs of every layer, using the source blobs' size.
2. `RunOnce` - performs all calculations on the source blob data.
3. `BackwardAndLearnOnce` - calculates the loss function gradient for all trainable weights and updates the trainable weights through backpropagation.

The learning process consists of many iterations, each calling `CDnn::RunAndLearnOnce` for new source data.

### Running the network

Sometimes during learning you will need to get the network response without changing the current parameters, for example, on test data for validation. In this case, use the `CDnn::RunOnce` method, which, unlike `CDnn::RunAndLearnOnce`, does not calculate the gradients and update the trainable parameters. This method is also used for working with the trained network.

## Serialization

Two classes are defined for serializing the network:

- `CArchiveFile` represents the file used for serialization
- `CArchive` represents the archive used to write and read from `CArchiveFile`

The serializing direction is determined by the settings with which the file and the archive instances are created: 

* to save the network into a file, create `CArchiveFile` with `CArchive::store` flag and an archive over it with `CArchive::SD_Storing` flag.
* to read the network from the file, use `CArchive::load` and `CArchive::SD_Loading` flags instead.

Once the archive has been created, call the [`CDnn::Serialize`](Dnn.md#serialization) method to serialize the network. The direction will be chosen automatically.

See also [more details about the classes used for serialization](../Common/README.md#serialization).

### Sample code for saving the network

```c++
CRandom random( 0x123 );
CDnn net( random, GetDefaultCpuMathEngine() );

/*
... Build and train the network ...
*/

CArchiveFile file( "my_net.archive", CArchive::store );
CArchive archive( &file, CArchive::SD_Storing );
archive.Serialize( net );
archive.Close();
file.Close();
```

## Using the network

```c++
// The math engine working on GPU that uses not more than 1GB GPU RAM
IMathEngine* gpuMathEngine = CreateGpuMathEngine( 1024 * 1024 * 1024, GetFmlExceptionHandler() );

{
    CRandom random( 0x123 );
    CDnn net( random, *gpuMathEngine );

    // Load the network
    {
      CArchiveFile file( "my_net.archive", CArchive::store );
      CArchive archive( &file, CArchive::SD_Storing );
      archive.Serialize( net );
      // file and archive will be closed in destructors
    }

    // The blob to store a single 32x32 RGB image
    CPtr<CDnnBlob> dataBlob = CDnnBlob::Create2DImageBlob( *gpuMathEngine, CT_Float, 1, 1, 32, 32, 3 );

    dataBlob->Fill( 0.5f ); // Filling with a constant value

    // Get the pointers to the source and the sink layers
    CPtr<CSourceLayer> src = CheckCast<CSourceLayer>( net.GetLayer( "source" ) );
    CPtr<CSinkLayer> sink = CheckCast<CSinkLayer>( net.GetLayer( "sink" ) );

    src->SetBlob( dataBlob ); // setting the input data
    net.RunOnce(); // running the network
    CPtr<CDnnBlob> resultBlob = sink->GetBlob(); // getting the response

    // Extract the data and put it in an array
    CArray<float> result;
    result.SetSize( resultBlob->GetDataSize() );
    resulBlob->CopyTo( result.GetPtr() );

    // Analyze the network response

    // Destroy all blobs and the network object
}

// Delete the engine after all blobs are deleted
delete gpuMathEngine;
```

## The layers

- [CBaseLayer](BaseLayer.md) is the base class for common layer functionality
- The layers used to pass the data to and from the network:
  - [CSourceLayer](IOLayers/SourceLayer.md) transmits a blob of data into the network
  - [CSinkLayer](IOLayers/SinkLayer.md) is used to retrieve a blob of data with the network response
  - [CProblemSourceLayer](IOLayers/ProblemSourceLayer.md) transmits the data from [`IProblem`](../ClassificationAndRegression/Problems.md) into the network
  - [CFullyConnectedSourceLayer](IOLayers/FullyConnectedSourceLayer.md) transmits the data from `IProblem` into the network, multiplying the vectors by a trainable weights matrix
- [CFullyConnectedLayer](FullyConnectedLayer.md) is the fully-connected layer
- [Activation functions](ActivationLayers/README.md):
  - [CLinearLayer](ActivationLayers/LinearLayer.md) - a linear activation function `ax + b`
  - [CELULayer](ActivationLayers/ELULayer.md) - `ELU` activation function
  - [CReLULayer](ActivationLayers/ReLULayer.md) - `ReLU` activation function
  - [CLeakyReLULayer](ActivationLayers/LeakyReLULayer.md) - `LeakyReLU` activation function
  - [CAbsLayer](ActivationLayers/AbsLayer.md) - `abs(x)` activation function
  - [CSigmoidLayer](ActivationLayers/SigmoidLayer.md) - `sigmoid` activation function
  - [CTanhLayer](ActivationLayers/TanhLayer.md) - `tanh` activation function
  - [CHardTanhLayer](ActivationLayers/HardTanhLayer.md) - `HardTanh` activation function
  - [CHardSigmoidLayer](ActivationLayers/HardSigmoidLayer.md) - `HardSigmoid` activation function
  - [CPowerLayer](ActivationLayers/PowerLayer.md) - `pow(x, exp)` activation function
  - [CHSwishLayer](ActivationLayers/HSwishLayer.md) - `h-swish` activation function
  - [CGELULayer](ActivationLayers/GELULayer.md) - `x * sigmoid(1.702 * x)` activation function
- Convolution layers:
  - [CConvLayer](ConvolutionLayers/ConvLayer.md) - 2-dimensional convolution
    - [CRleConvLayer](ConvolutionLayers/RleConvLayer.md) - convolution for 2-dimensional images in RLE format
  - [C3dConvLayer](ConvolutionLayers/3dConvLayer.md) - 3-dimensional convolution
  - [CTranposedConvLayer](ConvolutionLayers/TransposedConvLayer.md) - transposed 2-dimensional convolution
  - [C3dTranposedConvLayer](ConvolutionLayers/3dTransposedConvLayer.md) - transposed 3-dimensional convolution
  - [CChannelwiseConvLayer](ConvolutionLayers/ChannelwiseConvLayer.md) - channelwise convolution
  - [CTimeConvLayer](ConvolutionLayers/TimeConvLayer.md) - sequence convolution along the "time" axis
- Pooling layers:
  - [CMaxPoolingLayer](PoolingLayers/MaxPoolingLayer.md) - 2-dimensional max pooling
  - [CMeanPoolingLayer](PoolingLayers/MeanPoolingLayer.md) - 2-dimensional mean pooling
  - [C3dMaxPoolingLayer](PoolingLayers/3dMaxPoolingLayer.md) - 3-dimensional max pooling
  - [C3dMeanPoolingLayer](PoolingLayers/3dMeanPoolingLayer.md) - 3-dimensional mean pooling
  - [CGlobalMaxPoolingLayer](PoolingLayers/GlobalMaxPoolingLayer.md) - max pooling over whole objects
  - [CMaxOverTimePoolingLayer](PoolingLayers/MaxOverTimePoolingLayer.md) - max pooling over sequences along the "time" axis
- [CSoftmaxLayer](SoftmaxLayer.md) calculates softmax function
- [CDropoutLayer](DropoutLayer.md) implements random dropout
- [CBatchNormalizationLayer](BatchNormalizationLayer.md) implements batch normalization
- [CObjectNormalizationLayer](ObjectNormalizationLayer.md) implements normalization over the objects
- Elementwise operations with data blobs:
  - [CEltwiseSumLayer](EltwiseLayers/EltwiseSumLayer.md) - elementwise sum
  - [CEltwiseMulLayer](EltwiseLayers/EltwiseMulLayer.md) - elementwise product
  - [CEltwiseMaxLayer](EltwiseLayers/EltwiseMaxLayer.md) - elementwise maximum
  - [CEltwiseNegMulLayer](EltwiseLayers/EltwiseNegMulLayer.md) -calculates the elementwise product of `1 - first input` and the other inputs
- Auxiliary operations:
  - [CTransformLayer](TransformLayer.md) changes the blob shape
  - [CTransposeLayer](TransposeLayer.md) switches the blob dimensions
  - [CArgmaxLayer](ArgmaxLayer.md) finds maximum values along the given dimension
  - [CImageResizeLayer](ImageResizeLayer.md) changes the size of images in the blob
  - [CSubSequenceLayer](SubSequenceLayer.md) extracts subsequences
  - [CDotProductLayer](DotProductLayer.md) calculates the dot product of its inputs
  - [CAddToObjectLayer](AddToObjectLayer.md) - adds the content of one input to each of the objects of the other
  - [CMatrixMultiplicationLayer](MatrixMultiplicationLayer.md) - mutiplication of two sets of matrices
  - Blob concatenation:
    - [CConcatChannelsLayer](ConcatLayers/ConcatChannelsLayer.md) concatenates along the Channels dimension
    - [CConcatDepthLayer](ConcatLayers/ConcatDepthLayer.md) concatenates along the Depth dimension
    - [CConcatWidthLayer](ConcatLayers/ConcatWidthLayer.md) concatenates along the Width dimension
    - [CConcatHeightLayer](ConcatLayers/ConcatHeightLayer.md) concatenates along the Height dimension
    - [CConcatBatchWidthLayer](ConcatLayers/ConcatBatchWidthLayer.md) concatenates along the BatchWidth dimension
    - [CConcatObjectLayer](ConcatLayers/ConcatObjectLayer.md) concatenates the objects
  - Blbo splitting:
    - [CSplitChannelsLayer](SplitLayers/SplitChannelsLayer.md) splits along the Channels dimension
    - [CSplitDepthLayer](SplitLayers/SplitDepthLayer.md) splits along the Depth dimension
    - [CSplitWidthLayer](SplitLayers/SplitWidthLayer.md) splits along the Width dimension
    - [CSplitHeightLayer](SplitLayers/SplitHeightLayer.md) splits along the Height dimension
    - [CSplitBatchWidthLayer](SplitLayers/SplitBatchWidthLayer.md) splits along the BatchWidth dimension
  - Working with pixel lists:
    - [CPixelToImageLayer](PixelToImageLayer.md) creates images from the pixel lists
    - [CImageToPixelLayer](ImageToPixelLayer.md) extracts pixel lists from the images
  - Repeating data:
    - [CRepeatSequenceLayer](RepeatSequenceLayer.md) repeats sequences several times
    - [CUpsampling2DLayer](Upsampling2DLayer.md) scales up two-dimensional images
  - [CReorgLayer](ReorgLayer.md) transforms a multi-channel image into several smaller images with more channels
- Loss functions:
  - For binary classification:
    - [CBinaryCrossEntropyLossLayer](LossLayers/BinaryCrossEntropyLossLayer.md) - cross-entropy
    - [CHingeLossLayer](LossLayers/HingeLossLayer.md) - hinge loss function
    - [CSquaredHingeLossLayer](LossLayers/SquaredHingeLossLayer.md) - modified squared hinge loss function
    - [CBinaryFocalLossLayer](LossLayers/BinaryFocalLossLayer.md) - focal loss function (modified cross-entropy)
  - For multi-class classification:
    - [CCrossEntropyLossLayer](LossLayers/CrossEntropyLossLayer.md) - cross-entropy
    - [CMultiHingeLossLayer](LossLayers/MultiHingeLossLayer.md) - hinge loss function
    - [CMultiSquaredHingeLossLayer](LossLayers/MultiSquaredHingeLossLayer.md) - modified squared hinge loss function
    - [CFocalLossLayer](LossLayers/FocalLossLayer.md) - focal loss function (modified cross-entropy)
  - For regression:
    - [CEuclideanLossLayer](LossLayers/EuclideanLossLayer.md) - Euclidean distance
  - Additionally:
    - [CCenterLossLayer](LossLayers/CenterLossLayer.md) - the auxiliary *center loss* function that penalizes large variance inside a class
- Working with discrete features:
  - [CMultichannelLookupLayer](DiscreteFeaturesLayers/MultichannelLookupLayer.md) - vector representation of discrete features
  - [CAccumulativeLookupLayer](DiscreteFeaturesLayers/AccumulativeLookupLayer.md) - the sum of vector representations of a discrete feature
  - [CPositionalEmbeddingLayer](DiscreteFeaturesLayer/PositionalEmbeddingLayer.md) - the vector representations of a position in sequence
  - [CEnumBinarizationLayer](DiscreteFeaturesLayers/EnumBinarizationLayer.md) converts enumeration values to *one-hot encoding*
  - [CBitSetVectorizationLayer](DiscreteFeaturesLayers/BitSetVectorizationLayer.md) converts a *bitset* into a vector of ones and zeros
- Recurrent layers:
  - [CLstmLayer](LstmLayer.md) implements long short-term memory (LSTM)
  - [CGruLayer](GruLayer.md) implements a gated recurrent unit (GRU)
- [Conditional random field (CRF)](CrfLayers/README.md):
  - [CCrfLayer](CrfLayers/CrfLayer.md) represents a CRF
  - [CCrfLossLayer](CrfLayers/CrfLossLayer.md) calculates the loss function for training CRF
  - [CBestSequenceLayer](CrfLayers/BestSequenceLayer.md) finds optimal sequences in the results of CRF processing
- [Connectionist temporal classification (CTC)](CtcLayers/README.md):
  - [CCtcLossLayer](CtcLayers/CtcLossLayer.md) calculates the loss function
  - [CCtcDecodingLayer](CtcLayers/CtcDecodingLayer.md) finds the optimal sequences in CTC response
- Classification quality assessment:
  - [CAccuracyLayer](QualityControlLayers/AccuracyLayer.md) calculates the proportion of the objects classified correctly
  - [CPrecisionRecallLayer](QualityControlLayers/PrecisionRecallLayer.md) calculates the proportion of correctly classified objects for each of the two classes in binary classification
  - [CConfusionMatrixLayer](QualityControlLayers/ConfusionMatrixLayer.md) calculates the *confusion matrix* for multi-class classification
