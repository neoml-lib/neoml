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

#include <NeoML/NeoML.h>

namespace NeoML {

// Interface for setting input to a neural network
class IDistributedDataset {
public:
	virtual void SetInputBatch( CDnn& cnn, int thread ) = 0;
};

// Single process, multiple threads distributed training
class NEOML_API CDistributedTraining {
public:
	// Creates `count` cpu models
	explicit CDistributedTraining( CArchive& archive, int count );
	// Creates gpu models, `devs` should contain numbers of using devices
	explicit CDistributedTraining( CArchive& archive, const CArray<int>& cudaDevs );
	// Runs the networks, performs a backward pass and updates the trainable weights of all models
	void RunAndLearnOnce( IDistributedDataset& data );
	// Returns last loss of `layerName` for all models
	// `layerName` should correspond to CLossLayer or CCtcLossLayer
	void GetLastLoss( const CString& layerName, CArray<float>& losses );
	~CDistributedTraining();
protected:
	CArray<IMathEngine*> mathEngines;
	CArray<CRandom*> rands;
	CArray<CDnn*> cnns;

	void initialize( CArchive& archive, int count );
};


} // namespace NeoML