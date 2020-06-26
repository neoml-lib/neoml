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
#include <NeoML/Dnn/Layers/BatchNormalizationLayer.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/Dnn/Layers/FullyConnectedLayer.h>
#include <NeoML/Dnn/DnnSparseMatrix.h>
#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

// A fully-connected source layer, accepting the data from an object implementing the IProblem interface
// Similar to CProblemSourceLayer
class NEOML_API CFullyConnectedSourceLayer : public CFullyConnectedLayer {
	NEOML_DNN_LAYER( CFullyConnectedSourceLayer )
public:
	explicit CFullyConnectedSourceLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The number of vectors in the set on each iteration
	int GetBatchSize() const { return batchSize; }
	void SetBatchSize( int newBatchSize );

	// The maximum number of batches (vector sets) stored in math engine memory at the same time
	// 0 by default, which means that the whole problem is stored in memory
	int GetMaxBatchCount() const { return batchCount; }
	void SetMaxBatchCount( int newMaxBatchCount );

	// Retrieves or sets the original problem data
	// After the layer is connected to the network, you may change the problem 
	// only if the number of classes and features stays the same
	CPtr<const IProblem> GetProblem() const { return problem; }
	void SetProblem( IProblem* problem );

	// The data type for class labels
	TBlobType GetLabelType() const { return labelType; }
	void SetLabelType( TBlobType newLabelType );

protected:
	virtual ~CFullyConnectedSourceLayer();

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	CPtr<const IProblem> problem; // the current problem
	CDnnSparseMatrix* batchData; // the batch data
	CArray<float> batchLabels; // the correct labels for the vectors of the batch
	CArray<float> batchWeights; // the weights of the vectors of the batch
	int batchSize; // the number of vectors in the batch
	int batchCount; // the number of batches stored in math engine memory
	int batchIndex; // the index of the current batch
	int batchFirstLoadedIndex; // the index of the batch that was loaded first
	int batchLastLoadedIndex; // the index of the batch that was loaded last
	int firstVectorInBatchIndex; // the index of the first vector in the current batch
	TBlobType labelType; // the data type for class labels

	void loadBatchData();
	bool isBatchLoaded( int index ) const;
};

} // namespace NeoML
