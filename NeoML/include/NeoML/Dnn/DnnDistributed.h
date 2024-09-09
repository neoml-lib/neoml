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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/ArchiveFile.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoMLTest {
struct CDistributedTrainingTest;
}

namespace NeoML {

// Forward declaration
class CLoraSerializer;

// Interface for setting input to a neural network
class IDistributedDataset {
public:
	virtual ~IDistributedDataset() = default;
	// This method must set batches for all of the source layers in CDnn
	// Returns the current batch size (or 0, if there is no data for this thread on this run)
	// This batch size affects weights balance between different threads
	// Batch size doesn't affect different runs on the same thread (multiple RunAndBackwardOnce)
	// Batch size 0 isn't supported on the first run (because of CDnn initialization)
	virtual int SetInputBatch( CDnn& dnn, int thread ) = 0;
};

// Initializer to use in distributed training
enum class TDistributedInitializer {
	Xavier,
	XavierUniform,
	Uniform
};

//---------------------------------------------------------------------------------------------------------------------

// Single process, multiple threads distributed training
class NEOML_API CDistributedTraining {
public:
	// Creates `count` cpu models
	// If `count` is 0 or less, then the models number equal to the number of available CPU cores
	CDistributedTraining( const CDnn& dnn, int threadsCount,
		TDistributedInitializer initializer = TDistributedInitializer::Xavier, int seed = 42, size_t memoryLimit = 0 );
	CDistributedTraining( CArchive& archive, int threadsCount,
		TDistributedInitializer initializer = TDistributedInitializer::Xavier, int seed = 42, size_t memoryLimit = 0 );
	// Creates gpu models, `devs` should contain numbers of using devices
	CDistributedTraining( const CDnn& dnn, const CArray<int>& cudaDevs,
		TDistributedInitializer initializer = TDistributedInitializer::Xavier, int seed = 42, size_t memoryLimit = 0 );
	CDistributedTraining( CArchive& archive, const CArray<int>& cudaDevs,
		TDistributedInitializer initializer = TDistributedInitializer::Xavier, int seed = 42, size_t memoryLimit = 0 );

	virtual ~CDistributedTraining();

	// Gets the number of models in disitrbuted traning
	int GetModelCount() const { return cnns.Size(); }
	// Sets the solver for all of the models
	void SetSolver( CArchive& archive );
	// Sets the learning rate for all of the models
	void SetLearningRate( float rate );
	// Returns the current learning rate
	float GetLearningRate() const;
	// Runs the networks without backward and training
	// NOTE: Main thread waits while all tasks are done
	void RunOnce( IDistributedDataset& data );
	// Runs the networks and performs a backward pass
	// NOTE: Main thread waits while all tasks are done
	void RunAndBackwardOnce( IDistributedDataset& data );
	// Runs the networks, performs a backward pass and updates the trainable weights of all models
	// NOTE: Main thread waits while all tasks are done
	void RunAndLearnOnce( IDistributedDataset& data );
	// Updates the trainable weights of all models (after RunAndBackwardOnce)
	// NOTE: Main thread waits while all tasks are done
	void Train();
	// Returns last loss of `layerName` for all models
	// `layerName` should correspond to CLossLayer, CCtcLossLayer or CCrfLossLayer
	void GetLastLoss( const CString& layerName, CArray<float>& losses ) const;
	// Returns last blobs of `layerName` for all models
	// `layerName` should correspond to CSinkLayer
	// NOTE: This blobs are part of the dnn and may be overwritten by next task of 
	//       `RunOnce`, `RunAndBackwardOnce` or `RunAndLearnOnce`.
	//       Use it after each task and copy them if you need to store the result.
	void GetLastBlob( const CString& layerName, CObjectArray<const CDnnBlob>& blobs ) const;
	/// deprecated
	void GetLastBlob( const CString& layerName, CObjectArray<CDnnBlob>& blobs ) const;
	// Save trained net
	void Serialize( CArchive& archive );
	// Save the trained net with the given `index` with its solver state (optional)
	// An archive with solver state can later be passed to CDnn::SerializeCheckpoint to resume training
	void StoreDnn( CArchive& archive, int index, bool storeSolver );

private:
	struct CThreadParams;

	// Either multi-threads on a CPU or multi-devices GPU
	const bool isCpu;
	// If multi-threads on a CPU, it is an operator of worker threads
	IThreadPool* const threadPool;
	// Separate mathEngine for each thread or device both for CPU and GPU training
	// Cannot use CPointerArray, as CreateDistributedCpuMathEngines requires a raw array to initialize engines
	CArray<IMathEngine*> mathEngines;
	// Separate random generator for each dnn in a thread
	CPointerArray<CRandom> rands;
	// Separate dnn for each thread
	CPointerArray<CDnn> cnns;
	// Separate `batchSize` for each dnn (may be empty) in a thread
	CArray<int> batchSize;
	// `Train()` cannot be called if it `isFirstRun`
	// `batchSize` may not be equal 0, if it `isFirstRun` for `RunOnce`, `RunAndBackwardOnce` or `RunAndLearnOnce`.
	bool isFirstRun = true;
	// Containers for errors if it happened
	CArray<CString> errorMessages;

	void initialize( CArchive& archive, int count, TDistributedInitializer initializer, int seed );

	friend class CLoraSerializer;
	friend struct ::NeoMLTest::CDistributedTrainingTest;
};

//---------------------------------------------------------------------------------------------------------------------

// Single process, multiple threads distributed inference on CPU
class NEOML_API CDistributedInference {
public:
	// Creates `threadsCount` dnns for inference on CPU
	// If `threadsCount` is 0 or less, then the models number equal to the number of available CPU cores
	CDistributedInference( const CDnn& dnn, int threadsCount, bool optimizeDnn = true, size_t memoryLimit = 0 );
	CDistributedInference( CArchive& archive, int threadsCount, int seed = 42,
		bool optimizeDnn = true, size_t memoryLimit = 0 );

	virtual ~CDistributedInference();

	// Gets the created models number
	int GetModelCount() const { return threadPool->Size(); }
	// Runs the inference for all of the networks
	// NOTE: Main thread waits while all tasks are done
	void RunOnce( IDistributedDataset& data );
	// Returns last blobs of `layerName` for all models
	// `layerName` should correspond to CSinkLayer
	// NOTE: This blobs are part of the dnn and may be overwritten by next `RunOnce` task
	//       Use it after each `RunOnce` task and copy them if you need to stare the result.
	void GetLastBlob( const CString& layerName, CObjectArray<const CDnnBlob>& blobs ) const;

private:
	// Params to transfer to all threads function
	struct CThreadParams;

	// The operator of worker threads
	CPtrOwner<IThreadPool> threadPool;
	// Own CPU Math Engine
	CPtrOwner<IMathEngine> mathEngine;
	// Class to create reference dnns
	CPtr<CReferenceDnnFactory> referenceDnnFactory;
	// Each `RunOnce` task parameters
	CPtrOwner<CThreadParams> threadParams;
};

} // namespace NeoML
