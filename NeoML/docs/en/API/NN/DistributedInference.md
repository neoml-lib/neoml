# Class CDistributedInference

<!-- TOC -->

- [Class CDistributedInference](#class-cdistributedinference)
    - [Settings](#settings)
    - [Usage](#usage)

<!-- /TOC -->

This class implements neural network inference across multiple CPU threads simultaneously.
The number of copies of the passed neural network will equal the number of inference threads.
During network inference, each copy will not store its own set of trained layer parameters.
Instead, they will share common trained layer parameters among themselves,
thereby reducing memory consumption during inference.


## Settings

**1. Trained Neural Network for Inference**

   The network must be prepared for inference:

   * Extraneous layers necessary only for training must be removed.
   * All layers must have initialized weights, and the network must have been reshaped at least once.

   **1.1.** From program memory
   ```
   const CDnn& dnn
   ```
   **1.2.** From a file on disk
   ```
   CArchive& archive
   ```
   Additionally, if random number generator seed initialization is needed:
   ```
   int seed = 42
   ```
   Otherwise, the random number generator state will be taken directly from the passed neural network.

**2. Number of CPU Threads**

   Specifies the number of threads on which inference will run simultaneously.

   If set to 0, the number of threads will be determined based on the number of CPU cores of your computer.
   ```
   int threadsCount
   ```

**3. Network Architecture Optimization**

   Indicates whether to optimize the network architecture to speed up its inference.

   This optimization includes optimizations suitable only for CPUs.
   ```
   bool optimizeDnn = true
   ```

**4. Maximum Memory Usage**

   Specifies the maximum amount of memory that all networks together can use.

   If set to 0, no memory limit will be imposed, allowing inference to utilize available computer memory.
   ```
   size_t memoryLimit = 0
   ```

**5. Number of Models**

   Indicates the number of copies of the passed neural network, which will equal the number of threads.

   Each thread will execute exactly one neural network.
   ```
   int GetModelCount()
   ```

**6. Running Inference on All Networks Simultaneously**

   This method initiates inference on all networks simultaneously.
   ```
   void RunOnce( IDistributedDataset& data )
   ```
   The main thread from which this method is called will wait for all networks to complete their work.
   To execute inference, an object of the `IDistributedDataset` class is required, which sets necessary data inputs for each network according to the dataset.

**7. Obtaining Results**

   Results can be obtained in the form of an array of output data blobs.

   Since the next inference will overwrite the data in the blob memory, reading them is only possible between the current and next `RunOnce` calls.
   If these data are needed later, one needs to copy them from these blobs manually.
   ```
   void GetLastBlob( const CString& layerName, CObjectArray<const CDnnBlob>& blobs ) const
   ```


## Usage

To use this class, you'll need an object that will provide necessary data inputs for each network according to the dataset.

The example below demonstrates how this class sets the same input for all inference runs for all models in the form of an array of ones:
```cpp
class CCustomDataset : public IDistributedDataset {
public:
	CCustomDataset( int inputSize ) : inputSize( inputSize ) { input.Add( 1, inputSize ); }

	int SetInputBatch( CDnn& dnn, int /*thread_id*/ ) override {
		CPtr<CDnnBlob> in = CDnnBlob::CreateTensor( dnn.GetMathEngine(), CT_Float, { 1, 1, 1, 1, 1, 1, inputSize } );
		in->CopyFrom( input.GetPtr() );
		// To set inputs, you need to know their number, dimensions, and names in the network.
		CheckCast<CSourceLayer>( dnn.GetLayer( "data" ) )->SetBlob( in );
		// If the result > 0, inputs were set for this network, and inference would proceed.
		return 1;
	}

private:
	const int inputSize;
	CArray<float> input;
};
```

Example of executing multi-threaded inference:
```cpp
	// Assuming you have a trained network
	CRandom random( 42 );
	CDnn dnn( random, MathEngine() );

	// Create an object for multi-threaded inference for this network
	CDistributedInference distributed( dnn, /*count*/0 );

	{
		// Dataset for input feeding
		CCustomDataset dataset( inputSize );
		// Perform multi-threaded inference
		distributed.RunOnce( dataset );
	}

	{ // Retrieve results without copying
		CObjectArray<const CDnnBlob> results;
		distributed.GetLastBlob( "sink", results );
		// Apply the results for your purposes
		// ...
	}

	{
		// New dataset for input feeding
		CCustomDataset datasetNew( inputSize );
		// Perform multi-threaded inference again for new data
		distributed.RunOnce( datasetNew );
		// Apply the results again...
	}
```

A neural network can be read from an archive, and an object for multi-threaded inference
can be constructed with the necessary settings:
```cpp
	CDistributedInference distributed( archive,
		/*threadsCount*/ 4,
		/*seed*/ 42,
		/*optimizeDnn*/ false,
		/*memoryLimit*/ 300 * 1024 * 1024
	);
```
