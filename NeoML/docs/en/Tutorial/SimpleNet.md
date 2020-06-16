# Simple Network Sample

<!-- TOC -->

- [Simple Network Sample](#simple-network-sample)
	- [Creating the neural network object](#creating-the-neural-network-object)
	- [Creating the layers](#creating-the-layers)
	- [Creating the data blobs](#creating-the-data-blobs)
	- [Training the network](#training-the-network)
	- [Evaluating the results](#evaluating-the-results)
		- [Output](#output)

<!-- /TOC -->

This tutorial shows how to use **NeoML** to create and train a simple neural network that would perform classification of the well-known [MNIST](https://en.wikipedia.org/wiki/MNIST_database) data set.

## Creating the neural network object

A neural network is implemented by the [CDnn](../API/NN/Dnn.md) class. Before creating an instance, you need to create a math engine that will be used for calculations ([IMathEngine](../API/NN/MathEngine.md) interface) and a random numbers generator for all random values during layers' initialization and network training.

In this tutorial we will be working with the default math engine, which uses the CPU for calculations. But you can also use GPUs and set up additional parameters (see [IMathEngine](../API/NN/MathEngine.md) description for details).

```c++
// The default math engine calculating on CPU
IMathEngine& mathEngine = GetDefaultCpuMathEngine();
// Random number generator
CRandom random( 451 );
// Neural network
CDnn net( random, mathEngine );
```

## Creating the layers

Now we can create, set up, and connect the layers we need. Please note that the layers' names in the same network should be unique.

For the network solving a classification problem we will need:

1. Two [source layers](../API/NN/IOLayers/SourceLayer.md), the first one containing the images, the second — the correct class labels.
2. For classification we will put in a combination of a fully-connected layer and `ReLU` activation layer, twice.
3. On the output and the second input (the correct labels) calculate the loss function. The training will aim to minimize the loss function value.

This sample only trains the network to show that it is possible; the classification results are not displayed. To retrieve the results, you will need to connect a sink layer [CSinkLayer](../API/NN/IOLayers/SinkLayer.md) to the network output (in this case, the output of the `relu2` layer).

```c++
// The data input layer
CPtr<CSourceLayer> data = new CSourceLayer( mathEngine );
data->SetName( "data" ); // set the name unique for the network
net.AddLayer( *data ); // add the layer to the network

// The labels input layer
CPtr<CSourceLayer> label = new CSourceLayer( mathEngine );
label->SetName( "label" );
net.AddLayer( *label );

// The first fully-connected layer of size 1024
CPtr<CFullyConnectedLayer> fc1 = new CFullyConnectedLayer( mathEngine );
fc1->SetName( "fc1" );
fc1->SetNumberOfElements( 1024 ); // set the number of elements
fc1->Connect( *data ); // connect to the previous layer
net.AddLayer( *fc1 );

// The activation function
CPtr<CReLULayer> relu1 = new CReLULayer( mathEngine );
relu1->SetName( "relu1" );
relu1->Connect( *fc1 );
net.AddLayer( *relu1 );

// The second fully-connected layer of size 512
CPtr<CFullyConnectedLayer> fc2 = new CFullyConnectedLayer( mathEngine );
fc2->SetName( "fc2" );
fc2->SetNumberOfElements( 512 );
fc2->Connect( *relu1 );
net.AddLayer( *fc2 );

// The activation function
CPtr<CReLULayer> relu2 = new CReLULayer( mathEngine );
relu2->SetName( "relu2" );
relu2->Connect( *fc2 );
net.AddLayer( *relu2 );

// The third fully-connected layer of size equal to the number of classes (10)
CPtr<CFullyConnectedLayer> fc3 = new CFullyConnectedLayer( mathEngine );
fc3->SetName( "fc3" );
fc3->SetNumberOfElements( 10 );
fc3->Connect( *relu2 );
net.AddLayer( *fc3 );

// Cross-entropy loss function; this layer already calculates softmax 
// on its inputs, so there is no need to add a softmax layer before it 
CPtr<CCrossEntropyLossLayer> loss = new CCrossEntropyLossLayer( mathEngine );
loss->SetName( "loss" );
loss->Connect( 0, *fc3 ); // first input: the network response
loss->Connect( 1, *label ); // second input: the correct classes
net.AddLayer( *loss );
```

## Creating the data blobs

Let us put the input data into batches of 100 images each; the 60000-image data set will give us 600 iterations per learning epoch.

Use the [CDnnBlob](../API/NN/DnnBlob.md) class methods to create blobs and fill them with the input images and their class labels.

```c++
const int batchSize = 100; // the batch size
const int iterationPerEpoch = 600; // the training set contains 60000 images (600 batches)

// The data blob with 100 MNIST images (1 channel, 29 height and 28 width)
CPtr<CDnnBlob> dataBlob = CDnnBlob::Create2DImageBlob( mathEngine, CT_Float, 1, batchSize, 29, 28, 1 );
// The labels blob with 100 vectors of size 10, one-hot encoding the correct class label
CPtr<CDnnBlob> labelBlob = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, 1, batchSize, 10 );

// Transmitting the blobs into the source layer
data->SetBlob( dataBlob );
label->SetBlob( labelBlob );
```

## Training the network

To learn the network, call the `RunAndLearnOnce()` method of the network object. On each iteration, get the value of the loss function using the `GetLastLoss()` method of the loss layer and add up these values to keep track of total loss per learning epoch.

The input data is randomly reshuffled after each epoch.

```c++
for( int epoch = 1; epoch < 15; ++epoch ) {
    float epochLoss = 0; // total loss for the epoch
    for( int iter = 0; iter < iterationPerEpoch; ++iter ) {
        // trainData methods are used to transmit the data into the blob
        trainData.GetSamples( iter * batchSize, dataBlob );
        trainData.GetLabels( iter * batchSize, labelBlob );

        net.RunAndLearnOnce(); // run the learning iteration
        epochLoss += loss->GetLastLoss(); // add the loss value on the last step
    }

    ::printf( "Epoch #%02d    avg loss: %f\n", epoch, epochLoss / iterationPerEpoch );
    trainData.ReShuffle( random ); // reshuffle the data
}
```

## Evaluating the results

Test the trained network on the testing sample of 10000 images, also split into 100-image batches. To run the network without any training use the `RunOnce()` method. Again calculate the total loss over all the batches.

```c++
float testDataLoss = 0;
// The testing data set contains 10000 images (100 batches)
for( int testIter = 0; testIter < 100; ++testIter ) {
    testData.GetSamples( testIter * batchSize, dataBlob );
    testData.GetLabels( testIter * batchSize, labelBlob );
    net.RunOnce();
    testDataLoss += loss->GetLastLoss();
}

::printf( "\nTest data loss: %f\n", testDataLoss / 100 );
```

### Output

The test run of the network described above we received the following output:

```
Epoch #01    avg loss: 0.519273
Epoch #02    avg loss: 0.278983
Epoch #03    avg loss: 0.233433
Epoch #04    avg loss: 0.204021
Epoch #05    avg loss: 0.182192
Epoch #06    avg loss: 0.163927
Epoch #07    avg loss: 0.149121
Epoch #08    avg loss: 0.136408
Epoch #09    avg loss: 0.126139
Epoch #10    avg loss: 0.116643
Epoch #11    avg loss: 0.108768
Epoch #12    avg loss: 0.101515
Epoch #13    avg loss: 0.095355
Epoch #14    avg loss: 0.089328

Test data loss: 0.100225
```

You can see that the training set total loss is gradually decreasing with each training epoch. The testing set loss is also within the reasonable limits.
