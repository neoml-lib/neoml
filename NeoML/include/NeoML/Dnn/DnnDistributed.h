/* Copyright © 2017-2020 ABBYY Production LLC
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

namespace NeoML {

// Interface for setting input to a neural network
class IDistributedDataset {
public:
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

// Single process, multiple threads distributed training
class NEOML_API CDistributedTraining {
public:
	// Creates `count` cpu models
	CDistributedTraining( CDnn& dnn, int count,
		TDistributedInitializer initializer = TDistributedInitializer::Xavier, int seed = 42 );
	CDistributedTraining( CArchive& archive, int count,
		TDistributedInitializer initializer = TDistributedInitializer::Xavier, int seed = 42 );
	// Creates gpu models, `devs` should contain numbers of using devices
	CDistributedTraining( CDnn& dnn, const CArray<int>& cudaDevs,
		TDistributedInitializer initializer = TDistributedInitializer::Xavier, int seed = 42 );
	CDistributedTraining( CArchive& archive, const CArray<int>& cudaDevs,
		TDistributedInitializer initializer = TDistributedInitializer::Xavier, int seed = 42 );

	~CDistributedTraining();

	// Gets the number of models in disitrbuted traning
	int GetModelCount() const { return cnns.Size(); }
	// Sets the solver for all of the models
	void SetSolver( CArchive& archive );
	// Sets the learning rate for all of the models
	void SetLearningRate( float rate );
	// Runs the networks without backward and training
	void RunOnce( IDistributedDataset& data );
	// Runs the networks and performs a backward pass
	void RunAndBackwardOnce( IDistributedDataset& data );
	// Runs the networks, performs a backward pass and updates the trainable weights of all models
	void RunAndLearnOnce( IDistributedDataset& data );
	// Updates the trainable weights of all models (after RunAndBackwardOnce)
	void Train();
	// Returns last loss of `layerName` for all models
	// `layerName` should correspond to CLossLayer, CCtcLossLayer or CCrfLossLayer
	void GetLastLoss( const CString& layerName, CArray<float>& losses );
	// Returns last blobs of `layerName` for all models
	// `layerName` should correspond to CSinkLayer
	void GetLastBlob( const CString& layerName, CObjectArray<CDnnBlob>& blobs );
	// Save trained net
	void Serialize( CArchive& archive );

private:
	const bool isCpu;
	IThreadPool* threadPool;
	CArray<IMathEngine*> mathEngines;
	CArray<CRandom*> rands;
	CArray<CDnn*> cnns;
	CArray<int> batchSize;
	bool isFirstRun = true;
	CString errorMessage;

	void initialize( CArchive& archive, int count, TDistributedInitializer initializer, int seed );
};


} // namespace NeoML