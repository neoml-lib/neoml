# Class CReferenceDnnFactory

<!-- TOC -->

- [Class CReferenceDnnFactory](#class-creferencednnfactory)
    - [Reference DNN](#reference-dnn)
    - [Usage](#usage)

<!-- /TOC -->

This class can create copies of neural networks (reference dnn) based on the configuration of the original dnn it holds internally.
These reference dnn copies share weight parameters with the original dnn to save memory and to perform multi-threaded inference cache-friendly.
During multi-threaded inference, each thread independently operates on its own copy of the reference dnn.
Training and serialization are unavailable for both the original dnn and its reference dnn copies anymore.
Each reference dnn copy has its own random number generator, created by copying the current state of the original dnn's generator.
Each reference dnn uses its own generator during inference.
```cpp
class CReferenceDnnFactory {
public:
	// Archive should contain trained dnn, ready for inference
	// NOTE: mathEngine should be CPU only and live longer than CReferenceDnnFactory
	static CPtr<CReferenceDnnFactory> Create( IMathEngine& mathEngine, CArchive& archive,
		int seed = 42, bool optimizeDnn = true );
	// Dnn should be trained and ready for inference,
	// Dnn will be copied for internal storage to be non-disturbed, one can use argument dnn further as one wants
	// NOTE: mathEngine should be CPU only and live longer than CReferenceDnnFactory
	static CPtr<CReferenceDnnFactory> Create( IMathEngine& mathEngine, const CDnn& dnn, bool optimizeDnn = true );
	// Dnn should be trained and ready for inference, it will be moved inside and cannot be used outside.
	// NOTE: mathEngine should be CPU only and live longer than CReferenceDnnFactory
	static CPtr<CReferenceDnnFactory> Create( CDnn&& dnn, bool optimizeDnn = true );

	// Thread-safe coping of originalDnn, increments the counter
	// NOTE: The original dnn used to copy reference dnns may be also used as one more reference dnn (optimization)
	//       The 'getOriginDnn' flag must be used strictly for the only thread.
	CPtr<CDnnReference> CreateReferenceDnn( bool getOriginDnn = false );

protected:
	CPtr<CDnnReference> Origin; // The dnn to make reference dnns
};
```
This factory can be created from:
1. An archive containing a saved trained model of a neural network.
2. Copying a certain neural network already in memory (you can continue to work with the source dnn independently of reference dnns).
3. Capturing the state of the current neural network in memory (the state of the source network will be erased).

To obtain a new reference dnn, use the `CreateReferenceDnn` method of the factory with default parameters.
The original dnn is strictly one; it is necessary for the factory to create new reference dnn copies based on it.
Obtaining the original dnn is only for optimization purposes.
Never set the flag to retrieve the original network if you do not understand what you are doing.

Note that if `CPtr<CReferenceDnnFactory>` is no longer needed and the external pointer to it is cleared, it will only be destroyed in the destructor of the last reference dnn.
There are no shared pointer cycles here, as the original dnn, whose pointer is stored inside the factory, does not require the factory's existence.
If obtained, the original dnn may exist longer than the factory and may be destroyed when the last pointer to it is cleared, even after the destruction of the factory.


## Reference DNN

A reference dnn copy, which shares parameter weights, differs from a oridinary network (CDnn) in that it is embedded within a class:
```cpp
class CDnnReference {
public:
	CDnn Dnn;
protected:
	static CPtr<CDnnReference> Create( CRandom& random, IMathEngine& mathEngine );
};
```
This class can be obtained as a result from the `CreateReferenceDnn` method of the `CReferenceDnnFactory` class.

There is no need to pay attention to when `CPtr<CDnnReference>` will be cleared.

For the user, there is no difference between a reference dnn and the original dnn.
If there is no desire to use the original dnn for optimization, default parameters can simply be used.
The class `CDnnReference` is not part of the `CDnn` hierarchy.


## Usage

The general algorithm for multi-threaded inference can be illustrated with the following two code snippets.
```cpp
{
	// 1. Create and train the dnn 
	CDnn dnn( random, mathEngine );
	createAndTrainDnn( dnn );

	// 2. Prepare it for inference: delete label and loss layers, apply optimizations
	optimizeDnn( dnn );

	// 3. Create a factory corresponding of the given dnn
	CPtr<CReferenceDnnFactory> referenceDnnFactory = CReferenceDnnFactory::Create( mathEngine, dnn );

	// 4. Run multi-threaded inference
	CPtrOwner<IThreadPool> threadPool( CreateThreadPool( threadsCount ) );
	NEOML_NUM_THREADS( *threadPool, referenceDnnFactory.Ptr(), runDnnCreation )
}
```
We will need a trained neural network prepared for inference, a factory for creating reference dnns,
and a CPU thread pool to perform the inference workload. Tasks on it need to be started.

Additionally, we'll need a function performing inference within the thread pool:
```cpp
void runDnnCreation( int thread, void* arg )
{
	// 1. Create for given thread the necessary dnn
	CThreadsParams& params = *static_cast<CThreadsParams*>( arg );
	CPtr<CDnnReference> dnnRef = params.referenceDnnFactory->CreateReferenceDnn();

	// 2. Set input data for it and inference the dnn
	for( int i = 0; i < n; ++i ) {
		setInputDnn( dnnRef->Dnn  );
		dnn->RunOnce();

		CPtr<CDnnBlob> result = CheckCast<CSinkLayer>( dnn->GetLayer( "sink" ).Ptr() )->GetBlob();
		// 3. Apply results for your own work...
	}

	// 4. You may want to clear used buffers in pools for this thread after your work
	IMathEngine& mathEngine = dnn->GetMathEngine();
	dnnRef.Release();
	mathEngine.CleanUp();
}
```
In this case, for each thread independently, it creates a copy of the reference dnn.
It performs inference on the required data for its network.
Users can apply the results as they see fit.

