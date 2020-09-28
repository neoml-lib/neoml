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
#include <NeoML/Dnn/Dnn.h>
#include <NeoML/TraditionalML/Problem.h>
#include <NeoML/TraditionalML/Model.h>
#include <NeoML/TraditionalML/TrainingModel.h>

namespace NeoML {

class CSourceLayer;
class CSinkLayer;
class CDnnTrainingModelWrapper;

// CProblemSourceLayer is a wrapper over the IProblem interface. 
// On each iteration, it passes BatchSize vectors into the network for processing.
class NEOML_API CProblemSourceLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CProblemSourceLayer )
public:
	explicit CProblemSourceLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	int GetBatchSize() const { return batchSize; }
	void SetBatchSize(int _batchSize);

	// The filler for empty values that are not present in a sparse vector
	float GetEmptyFill() const { return emptyFill; }
	void SetEmptyFill(float _emptyFill) { NeoAssert(GetDnn() == 0); emptyFill = _emptyFill; }

	// You may only change the problem for the layer that is connected to a network
	// if the number of classes and the number of input vectors stay the same
	CPtr<const IProblem> GetProblem() const { return problem; }
	void SetProblem(const CPtr<const IProblem>& _problem);

	// Retrieves and sets the data type for class labels
	TBlobType GetLabelType() const { return labelType; }
	void SetLabelType( TBlobType newLabelType );

protected:
	virtual ~CProblemSourceLayer() {}

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	float emptyFill;		// the empty values filler (for values not represented in a sparse vector)
	int batchSize;			// the size of the batch passed to the network
	int nextProblemIndex;	// the index of the next element in the problem to be passed
	CPtr<const IProblem> problem;	// the classification problem the network is solving
	TBlobType labelType;		// the data type for labels
	CArray<float> exchangeBufs[3];
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CDnnModelWrapper is the base class wrapping the trained neural network into the IModel interface
class NEOML_API CDnnModelWrapper : public IModel {
public:
	explicit CDnnModelWrapper(IMathEngine& mathEngine, unsigned int seed = 0xDEADFACE);

	int GetClassCount() const override;
	bool Classify(const CSparseFloatVectorDesc& data, CClassificationResult& result) const override;
	bool Classify(const CFloatVector& data, CClassificationResult& result) const override;
	void Serialize(CArchive& archive) override;

protected:
	int ClassCount;
	float SourceEmptyFill;
	mutable CRandom Random;
	mutable CDnn Dnn;	// the network
	CPtr<CSourceLayer> SourceLayer;	// the reference to the source layer
	CPtr<CSinkLayer> SinkLayer;		// the reference to the terminator layer
	CPtr<CDnnBlob> SourceBlob;			// the source data blob
	mutable CArray<float> tempExp;		// the temporary array for exponent values to calculate softmax

	static const char* const SourceLayerName;
	static const char* const SinkLayerName;

	friend class CDnnTrainingModelWrapper;

private:
	IMathEngine& mathEngine;
	mutable CArray<float> exchangeBuffer;

	bool classify( CClassificationResult& result ) const;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CDnnTrainingModelWrapper is the base class wrapping the neural network 
// into an ITrainingModel interface so the network can be trained using the Train method
class NEOML_API CDnnTrainingModelWrapper : public ITrainingModel {
public:
	explicit CDnnTrainingModelWrapper( IMathEngine& _mathEngine ) : mathEngine( _mathEngine ) {}

	CPtr<IModel> Train(const IProblem& trainingClassificationData) override;

protected:
	// The function that builds and trains a neural network. 
	// The network should use the specified source and sink layers; 
	// the training data is passed to the problemLayer. 
	// None of these layers should be connected to any networks when the method is called.
	virtual void BuildAndTrainDnn(CDnn& cnn, const CPtr<CProblemSourceLayer>& problemLayer,
		const CPtr<CSourceLayer>& source, const CPtr<CSinkLayer>& sink) = 0;

private:
	IMathEngine& mathEngine;
};

} // namespace NeoML
