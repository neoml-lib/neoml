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
#include <NeoML/TraditionalML/FloatMatrix.h>
#include <NeoML/TraditionalML/ClassificationResult.h>
#include <NeoML/TraditionalML/TrainingModel.h>

namespace NeoML {

// The classification result for the one-versus-all classifier
// In addition to the regular classification results, contains the sigmoid sum of the binary classifiers ensemble
struct NEOML_API COneVersusAllClassificationResult : public CClassificationResult {
	// The sigmoid sum
	// You may use this to calculate the non-normalized probabilities returned by the binary classifiers
	double SigmoidSum;

	COneVersusAllClassificationResult() : CClassificationResult(), SigmoidSum( 1. ) {}
};

DECLARE_NEOML_MODEL_NAME( OneVersusAllModelName, "FmlOneVersusAllModel" )

// One-versus-all classification model interface
class NEOML_API IOneVersusAllModel : public IModel {
public:
	virtual ~IOneVersusAllModel();

	// Gets the basic IModel for all the binary classifiers
	virtual const CObjectArray<IModel>& GetModels() const = 0;

	// Gets the classification result with the info on normalized probabilities
	virtual bool ClassifyEx( const CSparseFloatVector& data, COneVersusAllClassificationResult& result ) const = 0;
	virtual bool ClassifyEx( const CSparseFloatVectorDesc& data, COneVersusAllClassificationResult& result ) const = 0;
	virtual bool ClassifyEx( const CFloatVector& data, COneVersusAllClassificationResult& result ) const = 0;
};

// One versus all classifier training interface
class NEOML_API COneVersusAll : public ITrainingModel {
public:
	explicit COneVersusAll( ITrainingModel& baseBinaryClassifier );

	// Sets a text stream for logging processing
	void SetLog( CTextStream* newLog ) { logStream = newLog; }

	// ITrainingModel interface methods:
	CPtr<IModel> Train( const ISparseClassificationProblem& trainingClassificationData ) override;
	CPtr<IModel> Train( const IDenseClassificationProblem& ) override;

private:
	ITrainingModel& baseBinaryClassifier; // the basic binary classifier used
	CTextStream* logStream; // the logging stream
};

} // namespace NeoML
